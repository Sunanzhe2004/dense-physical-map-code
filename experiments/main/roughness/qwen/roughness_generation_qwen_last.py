#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main pipeline: direct roughness generation from RGB plus an externally provided
# segmentation prior. This variant uses DashScope wan2.7-image instead of Ark.

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
DEFAULT_IMAGE_MODEL = "qwen-image-2.0"
DEFAULT_GENERATION_MODE = "edit"
DEFAULT_TIMEOUT = 1800
DEFAULT_WAN_OUTPUT_WIDTH = 1280
DEFAULT_WAN_OUTPUT_HEIGHT = 960
DEFAULT_WAN_OUTPUT_SIZE = f"{DEFAULT_WAN_OUTPUT_WIDTH}*{DEFAULT_WAN_OUTPUT_HEIGHT}"
WAN_MIN_PIXELS = 768 * 768
WAN_MAX_PIXELS = 2048 * 2048
PROMPT_VERSION = "capability_rgbonly_weakprior_v1_qwen_image_2"
SEG_CANDIDATE_SUFFIXES = ["_seg", "_mask", "_sam", "_semantic", "_label", ""]
INPUT_MODE_CHOICES = ("rgb_plus_seg", "rgb_only", "rgb_plus_example")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate roughness maps with wan2.7-image from RGB-only, RGB+example, or RGB+seg inputs."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="RGB image directory")
    parser.add_argument(
        "--seg_dir",
        type=str,
        default="",
        help="Segmentation map directory. Required only when --input_mode=rgb_plus_seg.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--input_mode",
        type=str,
        default="rgb_plus_seg",
        choices=INPUT_MODE_CHOICES,
        help="Input setting: original RGB+seg, RGB-only, or RGB+example.",
    )
    parser.add_argument("--example_rgb", type=str, default="", help="Reference RGB path for rgb_plus_example")
    parser.add_argument(
        "--example_roughness",
        type=str,
        default="",
        help="Reference roughness path for rgb_plus_example",
    )
    parser.add_argument(
        "--seg_suffix",
        type=str,
        default="",
        help="Preferred segmentation suffix, e.g. _seg or _mask. When set, it is matched before defaults.",
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
        help="Generation mode for wan2.7-image. edit is required for image-conditioned roughness generation.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=DEFAULT_WAN_OUTPUT_SIZE,
        help=f"Output size in WIDTH*HEIGHT format. Use adaptive to fall back to {DEFAULT_WAN_OUTPUT_SIZE}.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for generation")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--filename_suffix", type=str, default=None, help="Only process files ending with this suffix")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input_dir")
    parser.add_argument("--preserve_relative_dirs", action="store_true", help="Preserve paths relative to input_dir")
    parser.add_argument("--num_parts", type=int, default=1, help="Number of shards for parallel runs")
    parser.add_argument("--part_index", type=int, default=0, help="Shard index, 0-based")
    parser.add_argument(
        "--include_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit RGB filenames to process, e.g. 1_im.png 11_im.png 7_im.png",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing outputs. By default, completed outputs are skipped for resume.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Compatibility flag. Existing completed outputs are skipped by default unless --overwrite is set.",
    )
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


def validate_parts(num_parts: int, part_index: int) -> None:
    if num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if part_index < 0 or part_index >= num_parts:
        raise ValueError("--part_index must satisfy 0 <= part_index < --num_parts")


def shard_paths(paths: List[Path], num_parts: int, part_index: int) -> List[Path]:
    if num_parts == 1:
        return list(paths)
    return [path for idx, path in enumerate(paths) if idx % num_parts == part_index]


def list_images(input_dir: Path, filename_suffix: Optional[str] = None, recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    candidates = [p for p in sorted(iterator) if p.is_file()]

    if filename_suffix:
        images = [
            p for p in candidates
            if p.suffix.lower() in SUPPORTED_EXTS and p.name.endswith(filename_suffix)
        ]
        if images:
            return images
        recursive_msg = " recursively" if recursive else ""
        raise FileNotFoundError(
            f"No RGB images ending with {filename_suffix} were found{recursive_msg} in {input_dir}"
        )

    preferred = [
        p for p in candidates
        if p.suffix.lower() == ".png" and p.name.lower().endswith("_im.png")
    ]
    if preferred:
        return preferred

    image_prefix = [
        p for p in candidates
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if image_prefix:
        return image_prefix

    recursive_msg = " recursively" if recursive else ""
    raise FileNotFoundError(f"No RGB images matching *_im.png or Image_*.png were found{recursive_msg} in {input_dir}")


def filter_images_by_names(image_paths: List[Path], include_names: Optional[List[str]]) -> List[Path]:
    if not include_names:
        return image_paths

    wanted = [str(name).strip() for name in include_names if str(name).strip()]
    if not wanted:
        return image_paths

    image_map = {path.name: path for path in image_paths}
    missing = [name for name in wanted if name not in image_map]
    if missing:
        raise FileNotFoundError(f"Requested image(s) not found in input_dir: {missing}")
    return [image_map[name] for name in wanted]


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


def normalize_seg_suffix(suffix: str) -> str:
    text = str(suffix or "").strip().lower()
    if not text:
        return ""
    if text in {"none", "bare", "no_suffix", "nosuffix"}:
        return ""
    if not text.startswith("_"):
        text = "_" + text.lstrip("_")
    return text


def get_seg_candidate_suffixes(preferred_suffix: str = "") -> List[str]:
    preferred = normalize_seg_suffix(preferred_suffix)
    if not preferred:
        return list(SEG_CANDIDATE_SUFFIXES)
    suffixes = [preferred]
    suffixes.extend([suffix for suffix in SEG_CANDIDATE_SUFFIXES if suffix != preferred])
    return suffixes


def find_matching_seg(rgb_path: Path, seg_dir: Path, preferred_suffix: str = "") -> Path:
    stem = rgb_path.stem
    candidates: List[Path] = []
    seg_suffixes = get_seg_candidate_suffixes(preferred_suffix)

    for suffix in seg_suffixes:
        for ext in [rgb_path.suffix, ".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            candidates.append(seg_dir / f"{stem}{suffix}{ext}")

    name_lower = rgb_path.name.lower()
    if name_lower.startswith("image_"):
        seg_name = "ObjectSegmentation_" + rgb_path.name[len("Image_") :]
        candidates.append(seg_dir / seg_name)

    for tail in ["_im", "_rgb", "_image"]:
        if stem.endswith(tail):
            root = stem[: -len(tail)]
            for suffix in seg_suffixes:
                for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                    candidates.append(seg_dir / f"{root}{suffix}{ext}")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.resolve() == rgb_path.resolve():
                continue
        except Exception:
            if candidate == rgb_path:
                continue
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Cannot find segmentation image for {rgb_path.name} in {seg_dir}")


def build_roughness_prompt(input_mode: str) -> str:
    if input_mode == "rgb_only":
        return (
            "Generate a grayscale PBR roughness map from one RGB image, aligned to the input. "
            "Preserve layout and boundaries; do not add, remove, move, deform, duplicate, or hallucinate content. "
            "Black = low roughness = smooth, glossy, polished. White = high roughness = rough, matte, diffuse. Do not invert. "
            "Ranges: 0-.1 mirror/polished, .1-.3 smooth/glossy, .3-.6 semi-gloss to moderately rough, .6-.85 rough diffuse, .85-1 very rough. "
            "Do not copy brightness, lighting, shadows, shading, AO, reflections, highlights, or exposure gradients. "
            "Infer roughness from finish, highlight sharpness, reflection behavior, coating, and micro-structure, not brightness alone. "
            "Keep the same material similar across lighting. Prefer piecewise-smooth regions; local variation only when supported. "
            "Output only the roughness map."
        )

    if input_mode == "rgb_plus_example":
        return (
            "Input: (1) reference RGB, (2) reference roughness, (3) target RGB. "
            "Generate one roughness map for image (3), aligned to the target scene. "
            "Use image (2) only as format and tone reference, not as a layout template. "
            "Preserve target layout and boundaries; do not add, remove, move, deform, duplicate, or hallucinate content. "
            "Black = low roughness = smooth, glossy, polished. White = high roughness = rough, matte, diffuse. "
            "Ranges: 0-.1 mirror/polished, .1-.3 smooth/glossy, .3-.6 semi-gloss/moderately rough, .6-.85 rough diffuse, .85-1 very rough. "
            "Do not copy brightness, lighting, shadows, shading, reflections, highlights, or exposure gradients. "
            "Infer roughness from finish, reflection behavior, coating, and micro-structure. Keep material similar across lighting. "
            "Prefer stable smooth regions. Output only the roughness map."
        )

    return (
        "Input: (1) target RGB, (2) segmentation prior. "
        "Generate one roughness map aligned to the target RGB. "
        "Use segmentation only as a soft prior for boundaries and spatial stability, not as guaranteed material segmentation. "
        "Do not force one roughness value per segment; preserve RGB-supported material changes inside coarse masks. "
        "Preserve layout and boundaries; do not add, remove, move, deform, duplicate, or hallucinate content. "
        "Black = low roughness = smooth, glossy, polished. White = high roughness = rough, matte, diffuse. "
        "Ranges: 0-.1 mirror/polished, .1-.3 smooth/glossy, .3-.6 semi-gloss/moderately rough, .6-.85 rough diffuse, .85-1 very rough. "
        "Do not copy brightness, lighting, shadows, shading, reflections, highlights, or exposure gradients. "
        "Infer roughness from finish, reflection behavior, coating, and micro-structure. Output only the roughness map."
    )


def parse_wan_size(size_text: str) -> Tuple[int, int]:
    text = str(size_text or "").strip().lower()
    if not text or text == "adaptive":
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


def build_wan_size(size_text: str) -> str:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def upsert_manifest_entry(manifest: List[Dict[str, Any]], item: Dict[str, Any], key_field: str = "image_name") -> None:
    item_key = str(item.get(key_field, "") or item.get("image_name", "")).strip()
    if not item_key:
        manifest.append(item)
        return
    for idx, existing in enumerate(manifest):
        existing_key = str(existing.get(key_field, "") or existing.get("image_name", "")).strip()
        if existing_key == item_key:
            manifest[idx] = item
            return
    manifest.append(item)


def build_run_signature(
    *,
    input_mode: str,
    image_model: str,
    prompt_text: str,
    seg_suffix: str,
    example_rgb: str,
    example_roughness: str,
    timeout: int,
    size: str,
    generation_mode: str,
    watermark: bool,
    seed: int,
) -> Dict[str, Any]:
    if input_mode == "rgb_only":
        route = "wan_rgb_only"
    elif input_mode == "rgb_plus_example":
        route = "wan_rgb_plus_example"
    else:
        route = "wan_rgb_sam3_soft_direct"

    return {
        "image_model": image_model,
        "route": route,
        "input_mode": input_mode,
        "prompt_version": PROMPT_VERSION,
        "prompt_text": prompt_text,
        "segmentation_prior": "soft_spatial_prior" if input_mode == "rgb_plus_seg" else "unused",
        "segmentation_assumption": (
            "not_guaranteed_material_segmentation" if input_mode == "rgb_plus_seg" else "unused"
        ),
        "seg_suffix": seg_suffix,
        "example_rgb": example_rgb,
        "example_roughness": example_roughness,
        "timeout": timeout,
        "size": size,
        "generation_mode": generation_mode,
        "output_resolution_policy": "requested_fixed_output_size",
        "watermark": watermark,
        "seed": seed,
    }


def should_skip_existing_output(
    *,
    roughness_path: Path,
    output_meta_path: Path,
    run_signature: Dict[str, Any],
) -> bool:
    del output_meta_path, run_signature
    return is_completed_output(roughness_path)


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def build_part_path(meta_dir: Path, stem: str, ext: str, num_parts: int, part_index: int) -> Path:
    filename = f"{stem}.{ext}"
    if num_parts > 1:
        filename = f"{stem}.part{part_index + 1}of{num_parts}.{ext}"
    return meta_dir / filename


def build_image_output_path(
    image_path: Path,
    input_dir: Path,
    base_dir: Path,
    suffix: str,
    preserve_relative_dirs: bool,
) -> Path:
    if preserve_relative_dirs:
        relative_path = image_path.relative_to(input_dir)
        return base_dir / relative_path.parent / f"{image_path.stem}{suffix}"
    return base_dir / f"{image_path.stem}{suffix}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_seg_search_dir(
    rgb_path: Path,
    input_dir: Path,
    seg_dir: Path,
    preserve_relative_dirs: bool,
) -> Path:
    if not preserve_relative_dirs:
        return seg_dir
    relative_parent = rgb_path.relative_to(input_dir).parent
    candidate = seg_dir / relative_parent
    return candidate if candidate.exists() and candidate.is_dir() else seg_dir


def relative_output(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def get_pending_images(
    image_paths: List[Path],
    input_dir: Path,
    roughness_dir: Path,
    output_meta_dir: Path,
    run_signature: Dict[str, Any],
    overwrite: bool,
    preserve_relative_dirs: bool,
) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending = []
    for rgb_path in image_paths:
        roughness_path = build_image_output_path(
            rgb_path, input_dir, roughness_dir, "_roughness.png", preserve_relative_dirs
        )
        output_meta_path = build_image_output_path(
            rgb_path, input_dir, output_meta_dir, "_roughness.json", preserve_relative_dirs
        )
        if should_skip_existing_output(
            roughness_path=roughness_path,
            output_meta_path=output_meta_path,
            run_signature=run_signature,
        ):
            continue
        pending.append(rgb_path)
    return pending


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

    requested_size = build_wan_size(size)
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


def generate_roughness_map_with_wan(
    *,
    api_key: str,
    base_url: str,
    model: str,
    input_mode: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    save_path: Path,
    example_rgb: Optional[Path],
    example_roughness: Optional[Path],
    size: str,
    generation_mode: str,
    watermark: bool,
    seed: int,
    timeout: int,
) -> Dict[str, Any]:
    if input_mode == "rgb_only":
        prompt = build_roughness_prompt(input_mode)
        image_paths = [rgb_path]
        mode = "wan_rgb_only"
    elif input_mode == "rgb_plus_example":
        if not example_rgb or not example_roughness:
            raise ValueError("rgb_plus_example requires both --example_rgb and --example_roughness")
        prompt = build_roughness_prompt(input_mode)
        image_paths = [example_rgb, example_roughness, rgb_path]
        mode = "wan_rgb_plus_example"
    else:
        if seg_path is None:
            raise ValueError("rgb_plus_seg requires a matching segmentation image")
        prompt = build_roughness_prompt(input_mode)
        image_paths = [rgb_path, seg_path]
        mode = "wan_rgb_sam3_soft_direct"

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
        "roughness_mode": mode,
        "prompt_text": prompt,
        "request_id": response_json.get("request_id"),
        "image_url": image_url,
        "requested_size": build_wan_size(size),
        "input_size": list(get_image_size(rgb_path)),
        "returned_size": (response_json.get("usage") or {}).get("size"),
    }


def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key(args.api_key)

    input_dir = Path(args.input_dir)
    seg_dir = Path(args.seg_dir).expanduser() if args.seg_dir else None
    example_rgb = Path(args.example_rgb).expanduser() if args.example_rgb else None
    example_roughness = Path(args.example_roughness).expanduser() if args.example_roughness else None

    if args.input_mode == "rgb_plus_example":
        if not example_rgb or not example_roughness:
            raise ValueError("rgb_plus_example requires --example_rgb and --example_roughness")
        for path_obj, name in ((example_rgb, "example_rgb"), (example_roughness, "example_roughness")):
            if not path_obj.exists() or not path_obj.is_file():
                raise FileNotFoundError(f"{name} not found: {path_obj}")
    elif args.input_mode == "rgb_plus_seg":
        if seg_dir is None:
            raise ValueError("rgb_plus_seg requires --seg_dir")

    if seg_dir is not None and looks_like_object_seg_dir(input_dir):
        inferred_rgb_dir = infer_rgb_dir_from_seg_dir(input_dir)
        if inferred_rgb_dir is None:
            raise RuntimeError(
                f"input_dir looks like an ObjectSegmentation directory but the paired RGB directory could not be inferred: {input_dir}"
            )
        print(f"[info] detected segmentation directory as input_dir, switching RGB directory to: {inferred_rgb_dir}")
        seg_dir = input_dir
        input_dir = inferred_rgb_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roughness_dir = output_dir / "roughness"
    meta_dir = output_dir / "meta"
    output_meta_dir = meta_dir / "per_image"
    roughness_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths = filter_images_by_names(image_paths, args.include_names)
    image_paths_for_generate_all = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    image_paths_for_generate = shard_paths(image_paths_for_generate_all, args.num_parts, args.part_index)
    shard_label = f"part {args.part_index + 1}/{args.num_parts}"

    prompt_text = build_roughness_prompt(args.input_mode)
    run_signature = build_run_signature(
        input_mode=args.input_mode,
        image_model=args.image_model,
        prompt_text=prompt_text,
        seg_suffix=normalize_seg_suffix(args.seg_suffix),
        example_rgb=str(example_rgb) if example_rgb else "",
        example_roughness=str(example_roughness) if example_roughness else "",
        timeout=args.timeout,
        size=build_wan_size(args.size),
        generation_mode=args.generation_mode,
        watermark=args.watermark,
        seed=args.seed,
    )
    pending_image_paths = get_pending_images(
        image_paths_for_generate,
        input_dir,
        roughness_dir,
        output_meta_dir,
        run_signature,
        args.overwrite,
        args.preserve_relative_dirs,
    )

    print(
        f"[1/3] found {len(image_paths)} RGB images; "
        f"current shard {shard_label} owns {len(image_paths_for_generate)} images, "
        f"pending {len(pending_image_paths)} roughness maps with model={args.image_model}, "
        f"input_mode={args.input_mode}, generation_mode={args.generation_mode}"
    )
    print(
        "      "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"overwrite={args.overwrite} | "
        f"base_url={args.base_url}"
    )

    setup = dict(run_signature)
    setup.update(
        {
            "base_url": args.base_url,
            "filename_suffix": args.filename_suffix,
            "recursive": args.recursive,
            "preserve_relative_dirs": args.preserve_relative_dirs,
            "num_parts": args.num_parts,
            "part_index": args.part_index,
        }
    )
    setup_path = build_part_path(meta_dir, "setup", "json", args.num_parts, args.part_index)
    setup_path.write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = build_part_path(meta_dir, "manifest", "json", args.num_parts, args.part_index)
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)

    print(f"[2/3] start generation for {len(pending_image_paths)} pending images")
    if not pending_image_paths:
        print("  - current shard is already complete; exiting.")
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        relative_image_path = rgb_path.relative_to(input_dir).as_posix()
        roughness_path = build_image_output_path(
            rgb_path, input_dir, roughness_dir, "_roughness.png", args.preserve_relative_dirs
        )
        output_meta_path = build_image_output_path(
            rgb_path, input_dir, output_meta_dir, "_roughness.json", args.preserve_relative_dirs
        )
        item: Dict[str, Any] = {
            "image_name": rgb_path.name,
            "relative_image_path": relative_image_path,
            "prompt_version": PROMPT_VERSION,
            "input_mode": args.input_mode,
            "num_parts": args.num_parts,
            "part_index": args.part_index,
        }
        try:
            if not args.overwrite and should_skip_existing_output(
                roughness_path=roughness_path,
                output_meta_path=output_meta_path,
                run_signature=run_signature,
            ):
                item["skipped"] = True
                item["status"] = "skipped"
                item["skip_reason"] = "existing_completed_output"
                item["roughness_output"] = relative_output(roughness_path, output_dir)
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {relative_image_path} exists, skip")
                if not output_meta_path.exists():
                    ensure_parent_dir(output_meta_path)
                    output_meta = {
                        "image_name": rgb_path.name,
                        "relative_image_path": relative_image_path,
                        "roughness_output": relative_output(roughness_path, output_dir),
                        "input_mode": args.input_mode,
                        "run_signature": run_signature,
                        "recovered_existing_output": True,
                    }
                    output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                upsert_manifest_entry(manifest, item, key_field="relative_image_path")
                write_manifest(manifest_path, manifest)
                continue

            seg_path: Optional[Path] = None
            if args.input_mode == "rgb_plus_seg":
                if seg_dir is None:
                    raise ValueError("rgb_plus_seg requires --seg_dir")
                seg_search_dir = build_seg_search_dir(
                    rgb_path, input_dir, seg_dir, args.preserve_relative_dirs
                )
                seg_path = find_matching_seg(rgb_path, seg_search_dir, preferred_suffix=args.seg_suffix)
                item["seg_name"] = seg_path.name
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {relative_image_path} | seg={seg_path.name}")
            else:
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {relative_image_path}")

            ensure_parent_dir(roughness_path)
            ensure_parent_dir(output_meta_path)
            result = generate_roughness_map_with_wan(
                api_key=api_key,
                base_url=args.base_url,
                model=args.image_model,
                input_mode=args.input_mode,
                rgb_path=rgb_path,
                seg_path=seg_path,
                save_path=roughness_path,
                example_rgb=example_rgb,
                example_roughness=example_roughness,
                size=args.size,
                generation_mode=args.generation_mode,
                watermark=args.watermark,
                seed=args.seed,
                timeout=args.timeout,
            )
            item.update(result)
            item["status"] = "done"
            item["roughness_output"] = relative_output(roughness_path, output_dir)
            output_meta = {
                "image_name": rgb_path.name,
                "relative_image_path": relative_image_path,
                "roughness_output": relative_output(roughness_path, output_dir),
                "input_mode": args.input_mode,
                "run_signature": run_signature,
                "num_parts": args.num_parts,
                "part_index": args.part_index,
                "request_id": result.get("request_id"),
                "image_url": result.get("image_url"),
                "requested_size": result.get("requested_size"),
                "input_size": result.get("input_size"),
                "returned_size": result.get("returned_size"),
            }
            if seg_path is not None:
                output_meta["seg_name"] = seg_path.name
            output_meta_path.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(max(0.0, args.sleep))
        except Exception as e:
            item["error"] = str(e)
            item["status"] = "error"
            print(f"[error] {relative_image_path}: {e}")

        upsert_manifest_entry(manifest, item, key_field="relative_image_path")
        write_manifest(manifest_path, manifest)

    print("[3/3] done")
    print(f"roughness output dir: {roughness_dir.as_posix()}")
    print(f"meta output dir: {meta_dir.as_posix()}")
    print(f"output_dir: {output_dir.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
