#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Metallic generation with WAN2.7-Image from RGB-only, built-in final
# RGB+prompt, or RGB+example inputs. This script freezes the final
# surface_material_windowpane_filltight prompt variant.

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_IMAGE_MODEL = "wan2.7-image"
DEFAULT_GENERATION_MODE = "edit"
DEFAULT_TIMEOUT = 1800
DEFAULT_WAN_OUTPUT_WIDTH = 1280
DEFAULT_WAN_OUTPUT_HEIGHT = 720
DEFAULT_WAN_OUTPUT_SIZE = "same_as_input"
WAN_MIN_PIXELS = 768 * 768
WAN_MAX_PIXELS = 2048 * 2048
QWEN_PROMPT_MAX_CHARS = 800
WAN_PROMPT_MAX_CHARS = 5000
PROMPT_PRESET = "v3_visualprior_noboundary"
PROMPT_VERSION = "metallic_rgb_prompt_v3_visualprior_noboundary_wan2_7_surface_material_windowpane_filltight_final"
INPUT_MODE_CHOICES = ("rgb_only", "rgb_plus_prompt", "rgb_plus_example")
FINAL_SOFT_PRIOR_TEXT = (
    "This is an indoor scene. Predict a binary-like metallic map rather than a grayscale structure image. "
    "Default to non-metal black for most pixels. Glass and mirror remain black. Painted walls, ceramic, stone, wood, "
    "laminate, plastic, coated furniture, and painted board remain black unless there is clear exposed metal evidence. "
    "Do not preserve object boundaries using dark gray outlines. Only fill a broad object or housing region white when "
    "most of that visible surface clearly reads as exposed bare metal, such as an unmistakable stainless appliance face, "
    "exposed metal enclosure, or broad bare metal panel. Do not expand local metallic cues, bright reflections, or small "
    "metal parts into the rest of the object or housing. Small handles, hinges, knobs, and brackets may be white only when "
    "they visibly read as exposed metal. If evidence is ambiguous, choose black rather than gray."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metallic maps with WAN2.7-Image from RGB-only, RGB+prompt, or RGB+example inputs."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="RGB image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--input_mode",
        type=str,
        default="rgb_plus_prompt",
        choices=INPUT_MODE_CHOICES,
        help="Input setting: RGB-only baseline, RGB plus per-image prompt, or RGB+example pair.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="",
        help="Unused in the final script. The built-in final soft prior is used for --input_mode=rgb_plus_prompt.",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="_prompt.txt",
        help="Unused in the final script. Kept only for CLI compatibility.",
    )
    parser.add_argument("--example_rgb", type=str, default="", help="Reference RGB path for rgb_plus_example")
    parser.add_argument(
        "--example_metallic",
        type=str,
        default="",
        help="Reference metallic path for rgb_plus_example",
    )
    parser.add_argument("--api_key", type=str, default=None, help="DashScope API key")
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        help="DashScope base URL",
    )
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument(
        "--generation_mode",
        type=str,
        default=DEFAULT_GENERATION_MODE,
        choices=["generate", "edit"],
        help="Generation mode for wan2.7-image. edit is recommended for image-conditioned metallic generation.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=DEFAULT_WAN_OUTPUT_SIZE,
        help="Output size in WIDTH*HEIGHT format, or same_as_input / adaptive.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for generation")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist")
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = cli_api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing authentication: please provide --api_key or set DASHSCOPE_API_KEY.")
    return api_key


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def file_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def list_images(input_dir: Path) -> List[Path]:
    images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() == ".png"]
    if images:
        return images
    raise FileNotFoundError(f"No PNG images were found in {input_dir}")


def looks_like_object_seg_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("objectsegmentation_"):
            return True
    return False


def infer_rgb_dir_from_seg_dir(seg_dir: Path) -> Optional[Path]:
    parts = list(seg_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "objectsegmentation":
            parts[idx] = "Image"
            candidate = Path(*parts)
            if candidate.exists() and candidate.is_dir():
                return candidate
            break
    return None


def load_optional_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return text


def find_matching_prompt(rgb_path: Path, prompt_dir: Path, prompt_suffix: str) -> Path:
    suffix = str(prompt_suffix or "").strip()
    if not suffix:
        suffix = "_prompt.txt"

    candidates = [
        prompt_dir / f"{rgb_path.stem}{suffix}",
        prompt_dir / f"{rgb_path.name}{suffix}",
    ]
    if rgb_path.stem.lower().endswith("_im"):
        root = rgb_path.stem[:-3]
        candidates.append(prompt_dir / f"{root}{suffix}")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Cannot find prompt text for {rgb_path.name} in {prompt_dir}")


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def clamp_prompt(prompt: str, max_chars: int = QWEN_PROMPT_MAX_CHARS) -> str:
    prompt = compact_text(prompt)
    if len(prompt) <= max_chars:
        return prompt
    trimmed = prompt[: max(0, max_chars - 3)].rstrip(" ,;:")
    return trimmed + "..."


def get_prompt_char_limit(image_model: str) -> int:
    model = str(image_model or "").strip().lower()
    if "qwen-image-2.0" in model:
        return QWEN_PROMPT_MAX_CHARS
    if "wan2.7-image" in model:
        return WAN_PROMPT_MAX_CHARS
    return WAN_PROMPT_MAX_CHARS


def build_metallic_prompt(input_mode: str, scene_prompt: str = "", image_model: str = DEFAULT_IMAGE_MODEL) -> str:
    max_chars = get_prompt_char_limit(image_model)

    if input_mode == "rgb_plus_example":
        if max_chars <= QWEN_PROMPT_MAX_CHARS:
            prompt = (
                "Inputs: ref RGB, ref metallic, target RGB. Generate one aligned grayscale metallic map for the target only. "
                "Use the reference metallic only for output style, not layout. "
                "Black=non-metal/dielectric, white=clearly exposed metal; most indoor pixels should be pure black. "
                "Output only the metallic map, not a grayscale photo, relit image, or edge map. "
                "Do not copy brightness, shadows, reflections, highlights, AO, texture, or scene shading. "
                "Glass, mirror, wood, plastic, ceramic, stone, painted, or coated surfaces stay black unless exposed metal is clearly visible. "
                "Do not turn large smooth regions gray or white without clear exposed metal. "
                "Do not use gray to preserve boundaries, contour lines, reflective edges, decorative lines, or structure. "
                "If uncertain, choose black."
            )
        else:
            prompt = (
                "Inputs are: (1) reference RGB, (2) reference metallic map, (3) target RGB. "
                "Generate exactly one aligned grayscale metallic map for the target image (3) only. "
                "Use the reference metallic map only as output-style guidance for the meaning of black, gray, and white, "
                "not as a layout template, geometry template, or semantic template. "
                "Interpret the task as sparse metallic labeling, not grayscale scene rendering. "
                "Black means non-metal or dielectric. White means clearly exposed metallic material. "
                "Mid-gray is allowed only for genuinely ambiguous or partially metallic visible regions, but the default should remain black. "
                "Most indoor pixels should be pure black. Only relatively small exposed metal regions should become gray or white. "
                "The metallic map should be sparse; large contiguous gray/white regions are usually wrong. "
                "Output only the metallic map, not a grayscale photograph, not a relit image, not a desaturated scene image, not a shaded rendering, and not an edge map. "
                "Do not copy scene brightness, shadows, reflections, highlights, ambient occlusion, texture contrast, or object shading into the metallic map. "
                "Bright or white appearance alone is not evidence of metal. Smooth finish, glossy reflections, mirror-like highlights, ceramic-like highlights, porcelain, glazed ceramic, enamel-like surfaces, glossy paint, and smooth white fixtures should stay black unless exposed metal is clearly visible. "
                "Judge only the visible surface material, not the hidden substrate or what the object may be made of underneath paint or coating. "
                "Do not infer metal from object category alone: window frames, cabinets, doors, furniture, fixtures, and trim should default to near-black unless clearly visible bare metal is exposed. "
                "Treat transparent or translucent glass as non-metal by default; reflections or highlights on glass are not metal evidence. "
                "Do not brighten entire window panes, glass doors, outdoor scenery seen through glass, or room content visible behind glass; those broad glass-covered regions should stay near-black unless the visible surface itself is clearly exposed metal. "
                "As a conservative prior, treat glass as non-metal unless clearly exposed metallic surface is visible. "
                "Appliance fronts should stay near-black unless the visible face clearly looks like exposed bare stainless steel or brushed metal; only then is a larger contiguous metallic region allowed. "
                "Do not use metallic values to preserve object boundaries, panel boundaries, contour lines, reflective edges, decorative lines, silhouettes, or overall scene structure. "
                "Do not turn large smooth regions gray or white unless there is clear visible evidence of exposed metal. "
                "Do not fill walls, ceilings, floors, cabinet doors, or countertops with gray/white just to preserve shape; keep them near-black unless clear exposed bare metal is visible. "
                "Non-metal surfaces should stay black unless exposed metal is clearly visible. "
                "When uncertain, choose conservative near-black or black rather than preserving structure with gray."
            )
    else:
        if max_chars <= QWEN_PROMPT_MAX_CHARS:
            prompt = (
                "Input: one target RGB image. Generate one aligned grayscale metallic map only. "
                "Black=non-metal/dielectric, white=clearly exposed metal; most indoor pixels should be pure black. "
                "Output only the metallic map, not a grayscale photo, shaded scene, relit image, or edge map. "
                "Do not copy brightness, shadows, reflections, highlights, AO, texture, or scene shading. "
                "Glass, mirror, wood, plastic, ceramic, stone, painted, or coated surfaces stay black unless exposed metal is clearly visible. "
                "Do not turn large smooth regions gray or white without clear exposed metal. "
                "Do not use gray to preserve boundaries, contour lines, reflective edges, decorative lines, or structure in likely non-metal regions. "
                "If uncertain, choose black."
            )
        else:
            prompt = (
                "Input: one target RGB image. Generate exactly one aligned grayscale metallic map only. "
                "Interpret this as sparse metallic labeling rather than grayscale scene reconstruction. "
                "Black means non-metal or dielectric. White means clearly exposed metallic material. "
                "Gray should be used rarely and only for genuinely ambiguous or partially metallic visible regions. "
                "Most indoor pixels should be pure black. Only relatively small exposed metal regions should become gray or white. "
                "The metallic map should be sparse; large contiguous gray/white regions are usually wrong. "
                "Output only the metallic map, not a grayscale photo, not a desaturated image, not a shaded scene, not a relit result, and not an edge or contour map. "
                "Do not copy brightness, shadows, reflections, highlights, ambient occlusion, texture contrast, or scene shading into the metallic map. "
                "Bright or white appearance alone is not evidence of metal. Smooth finish, glossy reflections, mirror-like highlights, ceramic-like highlights, porcelain, glazed ceramic, enamel-like surfaces, glossy paint, and smooth white fixtures should stay black unless exposed metal is clearly visible. "
                "Judge only the visible surface material, not the hidden substrate or what the object may be made of underneath paint or coating. "
                "Do not infer metal from object category alone: window frames, cabinets, doors, furniture, fixtures, and trim should default to near-black unless clearly visible bare metal is exposed. "
                "Treat transparent or translucent glass as non-metal by default; reflections or highlights on glass are not metal evidence. "
                "Do not brighten entire window panes, glass doors, outdoor scenery seen through glass, or room content visible behind glass; those broad glass-covered regions should stay near-black unless the visible surface itself is clearly exposed metal. "
                "As a conservative prior, treat glass as non-metal unless clearly exposed metallic surface is visible. "
                "Appliance fronts should stay near-black unless the visible face clearly looks like exposed bare stainless steel or brushed metal; only then is a larger contiguous metallic region allowed. "
                "Do not use gray metallic values to preserve boundaries, silhouettes, contour lines, reflective edges, decorative lines, or overall object structure in likely non-metal regions. "
                "Do not turn large smooth surfaces gray or white without clear visible evidence of exposed metal. "
                "Do not fill walls, ceilings, floors, cabinet doors, or countertops with gray/white just to preserve shape; keep them near-black unless clear exposed bare metal is visible. "
                "Non-metal surfaces should stay black unless exposed metal is clearly visible. "
                "When uncertain, choose conservative near-black or black."
            )

    if scene_prompt:
        prefix = " Soft prior: "
        scene_prompt = compact_text(scene_prompt)
        budget = max(0, max_chars - len(prompt) - len(prefix))
        if budget > 0:
            prompt += prefix + scene_prompt[:budget]
    return clamp_prompt(prompt, max_chars=max_chars)


def parse_wan_size(size_text: str) -> Tuple[int, int]:
    text = str(size_text or "").strip().lower()
    if not text or text in {"adaptive", "same_as_input"}:
        return DEFAULT_WAN_OUTPUT_WIDTH, DEFAULT_WAN_OUTPUT_HEIGHT
    match = re.fullmatch(r"(\d+)\s*[*xX]\s*(\d+)", text)
    if not match:
        raise ValueError(f"Invalid size format: {size_text}. Expected WIDTH*HEIGHT, for example 1280*960.")
    return int(match.group(1)), int(match.group(2))


def validate_wan_output_size(width: int, height: int) -> None:
    pixels = width * height
    if pixels < WAN_MIN_PIXELS or pixels > WAN_MAX_PIXELS:
        raise ValueError(
            f"wan2.7-image output pixels must be within [{WAN_MIN_PIXELS}, {WAN_MAX_PIXELS}], got {width}x{height} ({pixels})."
        )
    ratio = width / float(height)
    if ratio < 1.0 / 8.0 or ratio > 8.0:
        raise ValueError(f"wan2.7-image output aspect ratio must be within [1:8, 8:1], got {width}x{height}.")


def build_wan_size(size_text: str, input_size: Optional[Tuple[int, int]] = None) -> str:
    text = str(size_text or "").strip().lower()
    if text == "same_as_input":
        if input_size is None:
            raise ValueError("same_as_input requires input_size.")
        width, height = input_size
    else:
        width, height = parse_wan_size(size_text)
    validate_wan_output_size(width, height)
    return f"{width}*{height}"


def summarize_http_error(response: requests.Response, max_len: int = 4000) -> str:
    try:
        body_text = json.dumps(response.json(), ensure_ascii=False)
    except Exception:
        body_text = response.text.strip()
    body_text = body_text.strip()
    if len(body_text) > max_len:
        body_text = body_text[:max_len] + "...(truncated)"
    reason = response.reason or ""
    return f"HTTP {response.status_code} {reason}: {body_text}" if body_text else f"HTTP {response.status_code} {reason}"


def extract_wan_image_url(response_json: Dict[str, Any]) -> str:
    for choice in response_json.get("output", {}).get("choices", []):
        message = choice.get("message", {})
        for item in message.get("content", []):
            if item.get("type") == "image" and item.get("image"):
                return item["image"]
            if item.get("image"):
                return item["image"]
    raise RuntimeError(f"Could not find image URL in model response: {json.dumps(response_json, ensure_ascii=False)}")


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_wan_image_response(response_json: Dict[str, Any], save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> str:
    image_url = extract_wan_image_url(response_json)
    save_url_to_file(image_url, save_path, timeout=timeout)
    return image_url


def enforce_grayscale_png(path: Path) -> None:
    Image.open(path).convert("L").save(path)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def write_manifest(path: Path, manifest: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def upsert_manifest_entry(manifest: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    image_name = str(item.get("image_name", "")).strip()
    if not image_name:
        manifest.append(item)
        return
    for idx, existing in enumerate(manifest):
        if str(existing.get("image_name", "")).strip() == image_name:
            manifest[idx] = item
            return
    manifest.append(item)


def build_run_signature(
    *,
    input_mode: str,
    image_model: str,
    prompt_text: str,
    prompt_dir: str,
    prompt_suffix: str,
    example_rgb: str,
    example_metallic: str,
    timeout: int,
    size: str,
    generation_mode: str,
    watermark: bool,
    seed: int,
) -> Dict[str, Any]:
    if input_mode == "rgb_plus_prompt":
        route = "wan_rgb_prompt_metallic"
    elif input_mode == "rgb_plus_example":
        route = "wan_rgb_example_metallic"
    else:
        route = "wan_rgb_only_metallic"

    return {
        "image_model": image_model,
        "route": route,
        "input_mode": input_mode,
        "prompt_version": PROMPT_VERSION,
        "prompt_preset": PROMPT_PRESET,
        "prompt_text": prompt_text,
        "prompt_dir": prompt_dir,
        "prompt_suffix": prompt_suffix,
        "example_rgb": example_rgb,
        "example_metallic": example_metallic,
        "timeout": timeout,
        "size": size,
        "generation_mode": generation_mode,
        "output_resolution_policy": "requested_fixed_output_size",
        "watermark": watermark,
        "seed": seed,
    }


def should_skip_existing_output(
    *,
    metallic_path: Path,
    output_meta_path: Path,
    run_signature: Dict[str, Any],
) -> bool:
    if not metallic_path.exists() or not output_meta_path.exists():
        return False
    saved = load_json_dict(output_meta_path)
    return saved.get("run_signature") == run_signature


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def run_wan_image_generation(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    image_paths: List[Path],
    generation_mode: str,
    size: str,
    watermark: bool,
    seed: int,
    timeout: int,
) -> Dict[str, Any]:
    mode = (generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")

    input_size = get_image_size(image_paths[-1]) if image_paths else None
    requested_size = build_wan_size(size, input_size=input_size)
    content: List[Dict[str, str]] = []
    if mode == "edit":
        for image_path in image_paths:
            content.append({"image": file_to_data_uri(image_path)})
    content.append({"text": prompt})

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "size": requested_size,
            "n": 1,
            "seed": seed,
            "watermark": watermark,
        },
    }
    response = requests.post(
        base_url.rstrip("/") + "/services/aigc/multimodal-generation/generation",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(summarize_http_error(response))
    response_json = response.json()
    if response_json.get("code"):
        raise RuntimeError(f"{response_json.get('code')}: {response_json.get('message', '')}")
    return response_json


def generate_metallic_map_with_wan(
    *,
    api_key: str,
    base_url: str,
    model: str,
    input_mode: str,
    rgb_path: Path,
    save_path: Path,
    scene_prompt: str,
    example_rgb: Optional[Path],
    example_metallic: Optional[Path],
    size: str,
    generation_mode: str,
    watermark: bool,
    seed: int,
    timeout: int,
) -> Dict[str, Any]:
    prompt = build_metallic_prompt(input_mode=input_mode, scene_prompt=scene_prompt, image_model=model)
    input_size = get_image_size(rgb_path)

    if input_mode == "rgb_only":
        image_paths = [rgb_path]
        mode = "wan_rgb_only_metallic"
    elif input_mode == "rgb_plus_example":
        if not example_rgb or not example_metallic:
            raise ValueError("rgb_plus_example requires both --example_rgb and --example_metallic")
        image_paths = [example_rgb, example_metallic, rgb_path]
        mode = "wan_rgb_example_metallic"
    else:
        image_paths = [rgb_path]
        mode = "wan_rgb_prompt_metallic"

    response_json = run_wan_image_generation(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        generation_mode=generation_mode,
        size=size,
        watermark=watermark,
        seed=seed,
        timeout=timeout,
    )
    image_url = save_wan_image_response(response_json, save_path, timeout=timeout)
    enforce_grayscale_png(save_path)
    return {
        "metallic_mode": mode,
        "prompt_text": prompt,
        "request_id": response_json.get("request_id"),
        "image_url": image_url,
        "requested_size": build_wan_size(size, input_size=input_size),
        "input_size": list(input_size),
        "returned_size": (response_json.get("usage") or {}).get("size"),
    }


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)

    input_dir = Path(args.input_dir)
    example_rgb = Path(args.example_rgb).expanduser() if args.example_rgb else None
    example_metallic = Path(args.example_metallic).expanduser() if args.example_metallic else None

    if looks_like_object_seg_dir(input_dir):
        inferred_rgb_dir = infer_rgb_dir_from_seg_dir(input_dir)
        if inferred_rgb_dir is None:
            raise RuntimeError(
                f"input_dir looks like an ObjectSegmentation directory but the paired RGB directory could not be inferred: {input_dir}"
            )
        print(f"[info] detected segmentation directory as input_dir, switching RGB directory to: {inferred_rgb_dir}")
        input_dir = inferred_rgb_dir

    if args.input_mode == "rgb_plus_example":
        if not example_rgb or not example_metallic:
            raise FileNotFoundError("rgb_plus_example requires --example_rgb and --example_metallic")
        for path_obj, name in ((example_rgb, "example_rgb"), (example_metallic, "example_metallic")):
            if not path_obj.exists() or not path_obj.is_file():
                raise FileNotFoundError(f"{name} not found: {path_obj}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metallic_dir = output_dir / "metallic"
    meta_dir = output_dir / "meta"
    output_meta_dir = meta_dir / "per_image"
    metallic_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths

    print(
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} metallic maps with model={args.image_model}, input_mode={args.input_mode}, generation_mode={args.generation_mode}"
    )

    prompt_text = build_metallic_prompt(
        args.input_mode,
        scene_prompt="[built-in final filltight soft prior]" if args.input_mode == "rgb_plus_prompt" else "",
        image_model=args.image_model,
    )
    signature_input_size = get_image_size(image_paths_for_generate[0]) if image_paths_for_generate else (DEFAULT_WAN_OUTPUT_WIDTH, DEFAULT_WAN_OUTPUT_HEIGHT)
    run_signature = build_run_signature(
        input_mode=args.input_mode,
        image_model=args.image_model,
        prompt_text=prompt_text,
        prompt_dir="",
        prompt_suffix="",
        example_rgb=str(example_rgb) if example_rgb else "",
        example_metallic=str(example_metallic) if example_metallic else "",
        timeout=args.timeout,
        size=build_wan_size(args.size, input_size=signature_input_size),
        generation_mode=args.generation_mode,
        watermark=args.watermark,
        seed=args.seed,
    )

    setup = dict(run_signature)
    setup["base_url"] = args.base_url
    (meta_dir / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = meta_dir / "manifest.json"
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)

    print(f"[2/3] start generation for {len(image_paths_for_generate)} images")
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        item: Dict[str, Any] = {"image_name": rgb_path.name}
        try:
            scene_prompt = ""
            if args.input_mode == "rgb_plus_prompt":
                scene_prompt = FINAL_SOFT_PRIOR_TEXT
                item["prompt_name"] = "builtin_filltight_soft_prior.txt"
                print(
                    f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name} | prompt=builtin_filltight_soft_prior.txt"
                )
            elif args.input_mode == "rgb_plus_example":
                item["example_rgb"] = example_rgb.name if example_rgb else ""
                item["example_metallic"] = example_metallic.name if example_metallic else ""
                print(
                    f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name} | example={example_rgb.name if example_rgb else ''}"
                )
            else:
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}")

            metallic_path = metallic_dir / f"{rgb_path.stem}_metallic.png"
            output_meta_path = output_meta_dir / f"{rgb_path.stem}_metallic.json"
            item["prompt_version"] = PROMPT_VERSION
            item["input_mode"] = args.input_mode
            if args.skip_existing and should_skip_existing_output(
                metallic_path=metallic_path,
                output_meta_path=output_meta_path,
                run_signature=run_signature,
            ):
                item["skipped"] = True
                item["skip_reason"] = "matching_output_and_signature"
                item["metallic_output"] = metallic_path.name
            else:
                result = generate_metallic_map_with_wan(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.image_model,
                    input_mode=args.input_mode,
                    rgb_path=rgb_path,
                    save_path=metallic_path,
                    scene_prompt=scene_prompt,
                    example_rgb=example_rgb,
                    example_metallic=example_metallic,
                    size=args.size,
                    generation_mode=args.generation_mode,
                    watermark=args.watermark,
                    seed=args.seed,
                    timeout=args.timeout,
                )
                item.update(result)
                item["metallic_output"] = metallic_path.name
                output_meta = {
                    "image_name": rgb_path.name,
                    "metallic_output": metallic_path.name,
                    "input_mode": args.input_mode,
                    "run_signature": run_signature,
                    "request_id": result.get("request_id"),
                    "image_url": result.get("image_url"),
                    "requested_size": result.get("requested_size"),
                    "input_size": result.get("input_size"),
                    "returned_size": result.get("returned_size"),
                }
                if args.input_mode == "rgb_plus_prompt":
                    output_meta["prompt_name"] = "builtin_filltight_soft_prior.txt"
                if args.input_mode == "rgb_plus_example":
                    output_meta["example_rgb"] = str(example_rgb) if example_rgb else ""
                    output_meta["example_metallic"] = str(example_metallic) if example_metallic else ""
                output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                time.sleep(max(0.0, args.sleep))
        except Exception as e:
            item["error"] = str(e)
            item["status"] = "error"
            print(f"[error] {rgb_path.name}: {e}")

        upsert_manifest_entry(manifest, item)
        write_manifest(manifest_path, manifest)

    print("[3/3] done")
    print(f"output_dir: {output_dir.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
