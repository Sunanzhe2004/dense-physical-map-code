#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main pipeline: direct relative depth generation from RGB only.
# Modified version:
# - near_white only (white=near, black=far)
# - no inversion in postprocess
# - unified prompt semantics across modes

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_IMAGE_MODEL = "wan2.7-image"
DEFAULT_GENERATION_MODE = "edit"
DEFAULT_IMAGE_SIZE = "1280*960"
DEFAULT_TIMEOUT = 1800
WAN_MIN_PIXELS = 768 * 768
WAN_MAX_PIXELS = 2048 * 2048
DEFAULT_WAN_OUTPUT_WIDTH = 1280
DEFAULT_WAN_OUTPUT_HEIGHT = 960
DEFAULT_WAN_FALLBACK_SIZE = f"{DEFAULT_WAN_OUTPUT_WIDTH}*{DEFAULT_WAN_OUTPUT_HEIGHT}"

PROMPT_VERSION = "relative_depth_benchmark_v9_near_white_only"
DEPTH_POLARITY_CHOICES = ("near_white",)
FIXED_DEPTH_POLARITY = "near_white"

INPUT_MODE_CHOICES = ("rgb_only", "rgb_plus_analysis", "rgb_plus_example")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate relative depth maps with Wan/DashScope from RGB inputs."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="RGB image directory")
    parser.add_argument(
        "--seg_dir",
        type=str,
        default="",
        help="Deprecated and ignored. Kept only for backward compatibility with older commands.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--input_mode",
        type=str,
        default="rgb_only",
        choices=INPUT_MODE_CHOICES,
        help="Strict RGB-only baseline or analysis/example-assisted baseline.",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="",
        help="Per-image scene-analysis text directory. Required only when --input_mode=rgb_plus_analysis.",
    )
    parser.add_argument(
        "--analysis_suffix",
        type=str,
        default="_analysis.txt",
        help="Per-image analysis suffix matched by RGB stem, e.g. *_im.png -> *_im_analysis.txt",
    )
    parser.add_argument("--example_rgb", type=str, default="", help="Reference RGB path for rgb_plus_example.")
    parser.add_argument(
        "--example_depth",
        type=str,
        default="",
        help="Reference depth path for rgb_plus_example. Must also be near_white (white=near, black=far).",
    )
    parser.add_argument(
        "--depth_polarity",
        type=str,
        default=FIXED_DEPTH_POLARITY,
        choices=DEPTH_POLARITY_CHOICES,
        help="Fixed grayscale convention: near_white only. White means near, black means far.",
    )
    parser.add_argument("--api_key", type=str, default=None, help="DashScope API key; defaults to DASHSCOPE_API_KEY")
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        help="DashScope base URL",
    )
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="DashScope image generation model")
    parser.add_argument(
        "--generation_mode",
        type=str,
        default=DEFAULT_GENERATION_MODE,
        choices=["generate", "edit"],
        help="generate: text-only generation; edit: attach input image(s), recommended for depth.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=DEFAULT_IMAGE_SIZE,
        help="Wan output size. source/original uses the target RGB size; explicit form: 1280*960.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for DashScope generation")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist")
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = cli_api_key or os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        api_key = api_key.strip()
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                "DASHSCOPE_API_KEY contains non-ASCII characters and cannot be written to the HTTP header. "
                "Please replace any placeholder text with the real key."
            ) from e
        return api_key
    raise RuntimeError("Missing authentication: please provide --api_key or DASHSCOPE_API_KEY.")


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


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def validate_wan_output_size(width: int, height: int) -> None:
    pixels = width * height
    if pixels < WAN_MIN_PIXELS or pixels > WAN_MAX_PIXELS:
        raise ValueError(
            f"wan2.7-image output pixels must be in [{WAN_MIN_PIXELS}, {WAN_MAX_PIXELS}], "
            f"got {width}x{height} ({pixels})."
        )
    ratio = width / float(height)
    if ratio < 1.0 / 8.0 or ratio > 8.0:
        raise ValueError(f"wan2.7-image output aspect ratio must be in [1:8, 8:1], got {width}x{height}.")


def normalize_wan_output_size(size: str, source_size: Tuple[int, int]) -> str:
    normalized = str(size or "").strip().lower()
    if not normalized or normalized in {"source", "original", "input", "auto"}:
        width, height = source_size
    elif normalized in {"adaptive", "2k"}:
        normalized = DEFAULT_WAN_FALLBACK_SIZE
        width, height = DEFAULT_WAN_OUTPUT_WIDTH, DEFAULT_WAN_OUTPUT_HEIGHT
    else:
        normalized = normalized.replace("x", "*")
        parts = normalized.split("*")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"wan2.7-image output size must look like 1280*960, got: {size}")
        width, height = int(parts[0]), int(parts[1])
    validate_wan_output_size(width, height)
    return f"{width}*{height}"


def list_images(input_dir: Path) -> List[Path]:
    preferred = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.name.lower().endswith("_im.png")
    ]
    if preferred:
        return preferred

    image_prefix = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if image_prefix:
        return image_prefix

    raise FileNotFoundError(f"No RGB images matching *_im.png or Image_*.png were found in {input_dir}")


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


def get_depth_semantics(depth_polarity: str = FIXED_DEPTH_POLARITY) -> Dict[str, Any]:
    normalized = str(depth_polarity or "").strip().lower()
    if normalized != FIXED_DEPTH_POLARITY:
        raise ValueError(
            f"This script only supports near_white depth polarity, got: {depth_polarity}"
        )

    return {
        "save_mode": FIXED_DEPTH_POLARITY,
        "prompt_mode": FIXED_DEPTH_POLARITY,
        "near_tone": "white",
        "far_tone": "black",
        "invert_after_save": False,
    }


def load_optional_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return text


def find_matching_analysis(rgb_path: Path, analysis_dir: Path, analysis_suffix: str) -> Path:
    suffix = str(analysis_suffix or "").strip()
    if not suffix:
        suffix = "_analysis.txt"

    candidates = [
        analysis_dir / f"{rgb_path.stem}{suffix}",
        analysis_dir / f"{rgb_path.name}{suffix}",
    ]
    if rgb_path.stem.lower().endswith("_im"):
        root = rgb_path.stem[:-3]
        candidates.append(analysis_dir / f"{root}{suffix}")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Cannot find analysis text for {rgb_path.name} in {analysis_dir}")


def build_relative_depth_prompt(
    depth_polarity: str,
    analysis_text: str = "",
    input_mode: str = "rgb_only",
) -> str:
    semantics = get_depth_semantics(depth_polarity)
    near_tone = semantics["near_tone"]
    far_tone = semantics["far_tone"]

    common_body = (
    "This is a relative depth prediction task, not metric depth regression. "
    "Predict camera-centric scene depth up to an unknown affine transform. "
    "The output must be a single-channel grayscale depth-buffer-like map, pixel-aligned with the target RGB image. "
    "Do not crop, zoom, shift, recompose, add, remove, or move any scene content. "

    f"{near_tone.capitalize()} means closer to the camera. {far_tone.capitalize()} means farther from the camera. "
    "Use a white near tone for close surfaces and a black far tone for distant surfaces. "

    "Infer the 3D layout first, then render only the depth map. "
    "Use geometric cues such as perspective, occlusion ordering, contact and support relations, object placement, surface orientation, room layout, and relative size. "
    "Occluding surfaces must be closer than the surfaces they occlude. "

    "Completely ignore photometric appearance when assigning depth: brightness, darkness, shadows, highlights, reflections, exposure, albedo, color, texture, material, gloss, image noise, and local contrast must not change the predicted depth. "
    "A bright patch on a floor, wall, door, cabinet, or tabletop remains at that surface's geometric depth. "
    "A dark shadow or dark material does not make a region farther away. "
    "Never make the result look like ambient occlusion, relighting, shading, or a grayscale rendering. "

    "Large planar or near-planar surfaces such as floors, walls, ceilings, doors, cabinets, and tabletops should be smooth and perspective-consistent. "
    "Within each visible surface face, depth should change only according to geometry and perspective, not according to illumination gradients or material texture. "
    "Reserve strong contrast mainly for true depth discontinuities: silhouettes, occlusion boundaries, contact edges, and object-to-background transitions. "

    "For openings, windows, mirrors, glass, and overexposed regions, estimate depth from the actual visible geometry. "
    "If a finite surface is visible, use that surface's depth. "
    "If the region is an opening with no visible finite surface, assign it far-background depth. "

    "Thin structures such as chair legs, table legs, lamp cords, plant stands, and object edges should remain coherent and aligned, but should not introduce texture-like grayscale noise. "
    "Use the full image-level depth range consistently: nearest valid visible surfaces should be close to white, farthest valid visible regions should be close to black, and intermediate surfaces should preserve correct ordinal ordering. "

    "The final image must be a clean training-label-style depth map, not a visually pleasing grayscale image. "
    "Do not stylize. Do not add text, labels, color, collage, borders, or overlays. "
    "Output exactly one grayscale depth map only."
)

    if input_mode == "rgb_plus_example":
        prompt = (
            "You are a senior monocular depth estimation and 3D scene geometry expert. "
            "You are given exactly three input images in order: "
            "(1) reference RGB image, "
            "(2) reference relative depth map, "
            "(3) target RGB image. "
            "Generate only the dense relative depth map for image (3). "
            "Use image (2) only to infer: "
            "the polarity convention that white means near and black means far, "
            "the clean grayscale depth-label style, "
            "and the desired degree of piecewise-smoothness. "
            "Do not copy or transfer the reference scene layout, object placement, object geometry, room geometry, global brightness histogram, contrast distribution, depth range compression, local shading pattern, edge softness, or scene-specific tonal balance onto the target image. "
            "The reference depth map uses the same polarity convention as the desired output: white=near, black=far. "
            "The target depth distribution must be determined only from the target RGB image's geometry. "
            "Do not normalize the target output to imitate the reference depth map's overall brightness distribution or contrast profile. "
            "If the reference depth map and the target RGB image suggest different depth distributions, follow the target RGB image only. "
            + common_body
        )
    else:
        prompt = (
            "You are a senior monocular depth estimation and 3D scene geometry expert. "
            "You are given exactly one input image: the target RGB image. "
            "Generate only a dense single-image relative depth map for that RGB image. "
            + common_body
        )

    if analysis_text:
        prompt += (
            " Additional scene analysis is provided below as a weak reasoning prior. "
            "Use it only when it is consistent with the RGB image, and never let it override visible geometry evidence. "
            "If the analysis conflicts with the visible scene structure, follow the RGB image only. "
            "Scene analysis: "
            f"{analysis_text} "
        )

    return prompt


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
    raise RuntimeError(f"Model response contains no image URL: {json.dumps(response_json, ensure_ascii=False)}")


def run_wan_image_generation(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    image_paths: List[Path],
    generation_mode: str,
    size: str,
    watermark: bool,
    seed: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Dict[str, Any], str, List[Tuple[int, int]]]:
    mode = (generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    input_sizes = [get_image_size(path) for path in image_paths]
    requested_size = normalize_wan_output_size(size, input_sizes[-1])
    content: List[Dict[str, Any]] = []
    if mode == "edit":
        content.extend({"image": file_to_data_uri(path)} for path in image_paths)
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
    return response_json, requested_size, input_sizes


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with save_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_wan_image_response(response_json: Dict[str, Any], save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> str:
    image_url = extract_wan_image_url(response_json)
    save_url_to_file(image_url, save_path, timeout=timeout)
    return image_url


def postprocess_depth_png(path: Path, depth_polarity: str, final_size: Tuple[int, int]) -> None:
    # Near-white only; no inversion is ever applied.
    del depth_polarity
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != final_size:
            image = image.resize(final_size, Image.Resampling.LANCZOS)
        image.save(path)


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
    base_url: str,
    image_model: str,
    prompt_text: str,
    depth_polarity: str,
    analysis_dir: str,
    analysis_suffix: str,
    example_rgb: str,
    example_depth: str,
    timeout: int,
    generation_mode: str,
    size: str,
    seed: int,
    watermark: bool,
) -> Dict[str, Any]:
    if input_mode == "rgb_plus_analysis":
        route = "wan_rgb_relative_depth_analysis"
    elif input_mode == "rgb_plus_example":
        route = "wan_rgb_relative_depth_example"
    else:
        route = "wan_rgb_relative_depth_direct"
    return {
        "image_model": image_model,
        "base_url": base_url,
        "route": route,
        "input_mode": input_mode,
        "prompt_version": PROMPT_VERSION,
        "prompt_text": prompt_text,
        "depth_representation": "relative_depth",
        "depth_polarity": depth_polarity,
        "analysis_dir": analysis_dir,
        "analysis_suffix": analysis_suffix,
        "example_rgb": example_rgb,
        "example_depth": example_depth,
        "evaluation_protocol": "single_image_relative_depth_prediction",
        "evaluation_note": "Near-white only. No polarity inversion, scale fit, or shift alignment is applied during generation.",
        "timeout": timeout,
        "generation_mode": generation_mode,
        "size": size,
        "seed": seed,
        "padding_mode": "none",
        "output_resolution_policy": "direct_source_image_input_final_source_size",
        "watermark": watermark,
    }


def should_skip_existing_output(
    *,
    depth_path: Path,
    output_meta_path: Path,
    run_signature: Dict[str, Any],
) -> bool:
    if not depth_path.exists() or not output_meta_path.exists():
        return False
    saved = load_json_dict(output_meta_path)
    return saved.get("run_signature") == run_signature


def generate_relative_depth_map_with_wan(
    *,
    api_key: str,
    base_url: str,
    model: str,
    generation_mode: str,
    input_mode: str,
    rgb_path: Path,
    save_path: Path,
    size: str,
    watermark: bool,
    seed: int,
    timeout: int,
    depth_polarity: str,
    analysis_text: str,
    example_rgb: Optional[Path],
    example_depth: Optional[Path],
) -> Dict[str, Any]:
    prompt = build_relative_depth_prompt(
        depth_polarity,
        analysis_text=analysis_text,
        input_mode=input_mode,
    )
    if input_mode == "rgb_plus_example":
        if not example_rgb or not example_depth:
            raise ValueError("rgb_plus_example requires both --example_rgb and --example_depth")
        image_paths = [example_rgb, example_depth, rgb_path]
        mode = "wan_rgb_relative_depth_example"
    else:
        image_paths = [rgb_path]
        mode = "wan_rgb_relative_depth_analysis" if analysis_text else "wan_rgb_relative_depth_direct"
    source_size = get_image_size(rgb_path)
    response_json, requested_size, input_sizes = run_wan_image_generation(
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
    postprocess_depth_png(save_path, depth_polarity=depth_polarity, final_size=source_size)
    usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
    return {
        "mode": mode,
        "image_url": image_url,
        "requested_size": requested_size,
        "input_sizes": [list(size_pair) for size_pair in input_sizes],
        "source_size": list(source_size),
        "final_size": list(source_size),
        "returned_size": usage.get("size"),
        "request_id": response_json.get("request_id") if isinstance(response_json, dict) else None,
    }


def main() -> None:
    args = parse_args()

    if args.depth_polarity != FIXED_DEPTH_POLARITY:
        raise ValueError(
            f"This script only supports {FIXED_DEPTH_POLARITY}, got: {args.depth_polarity}"
        )

    api_key = ensure_api_key(args.api_key)

    input_dir = Path(args.input_dir)
    analysis_dir = Path(args.analysis_dir).expanduser() if args.analysis_dir else None
    example_rgb = Path(args.example_rgb).expanduser() if args.example_rgb else None
    example_depth = Path(args.example_depth).expanduser() if args.example_depth else None

    if looks_like_object_seg_dir(input_dir):
        inferred_rgb_dir = infer_rgb_dir_from_seg_dir(input_dir)
        if inferred_rgb_dir is None:
            raise RuntimeError(
                f"input_dir looks like an ObjectSegmentation directory but the paired RGB directory could not be inferred: {input_dir}"
            )
        print(f"[info] detected segmentation directory as input_dir, switching RGB directory to: {inferred_rgb_dir}")
        input_dir = inferred_rgb_dir

    if args.seg_dir:
        print("[info] --seg_dir is deprecated for depth_generation.py and will be ignored.")

    if args.input_mode == "rgb_plus_analysis":
        if analysis_dir is None or not analysis_dir.exists() or not analysis_dir.is_dir():
            raise FileNotFoundError("rgb_plus_analysis requires a valid --analysis_dir")
    elif args.input_mode == "rgb_plus_example":
        if not example_rgb or not example_depth:
            raise FileNotFoundError("rgb_plus_example requires --example_rgb and --example_depth")
        for path_obj, name in ((example_rgb, "example_rgb"), (example_depth, "example_depth")):
            if not path_obj.exists() or not path_obj.is_file():
                raise FileNotFoundError(f"{name} not found: {path_obj}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = output_dir / "relative_depth"
    meta_dir = output_dir / "meta"
    output_meta_dir = meta_dir / "per_image"
    depth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths

    print(
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} relative depth maps with model={args.image_model}, input_mode={args.input_mode}, depth_polarity={args.depth_polarity}"
    )

    prompt_text = build_relative_depth_prompt(
        args.depth_polarity,
        analysis_text="[per-image analysis text inserted at runtime]" if args.input_mode == "rgb_plus_analysis" else "",
        input_mode=args.input_mode,
    )
    run_signature = build_run_signature(
        input_mode=args.input_mode,
        base_url=args.base_url,
        image_model=args.image_model,
        prompt_text=prompt_text,
        depth_polarity=args.depth_polarity,
        analysis_dir=str(analysis_dir) if analysis_dir else "",
        analysis_suffix=args.analysis_suffix,
        example_rgb=str(example_rgb) if example_rgb else "",
        example_depth=str(example_depth) if example_depth else "",
        timeout=args.timeout,
        generation_mode=args.generation_mode,
        size=args.size,
        seed=args.seed,
        watermark=args.watermark,
    )

    setup = dict(run_signature)
    (meta_dir / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = meta_dir / "manifest.json"
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)

    print(f"[2/3] start generation for {len(image_paths_for_generate)} images")
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        item: Dict[str, Any] = {"image_name": rgb_path.name}
        try:
            analysis_path: Optional[Path] = None
            analysis_text = ""
            if args.input_mode == "rgb_plus_analysis":
                if analysis_dir is None:
                    raise FileNotFoundError("rgb_plus_analysis requires --analysis_dir")
                analysis_path = find_matching_analysis(rgb_path, analysis_dir, args.analysis_suffix)
                analysis_text = load_optional_text(analysis_path)
                if not analysis_text:
                    raise RuntimeError(f"Analysis text is empty: {analysis_path}")
                item["analysis_name"] = analysis_path.name
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name} | analysis={analysis_path.name}")
            elif args.input_mode == "rgb_plus_example":
                item["example_rgb"] = example_rgb.name if example_rgb else ""
                item["example_depth"] = example_depth.name if example_depth else ""
                print(
                    f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name} | example={example_rgb.name if example_rgb else ''}"
                )
            else:
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}")

            depth_path = depth_dir / f"{rgb_path.stem}_relative_depth.png"
            output_meta_path = output_meta_dir / f"{rgb_path.stem}_relative_depth.json"
            item["prompt_version"] = PROMPT_VERSION
            item["input_mode"] = args.input_mode
            item["depth_representation"] = "relative_depth"
            item["depth_polarity"] = args.depth_polarity

            if args.skip_existing and should_skip_existing_output(
                depth_path=depth_path,
                output_meta_path=output_meta_path,
                run_signature=run_signature,
            ):
                item["skipped"] = True
                item["skip_reason"] = "matching_output_and_signature"
                item["relative_depth_output"] = depth_path.name
            else:
                generation_result = generate_relative_depth_map_with_wan(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.image_model,
                    generation_mode=args.generation_mode,
                    input_mode=args.input_mode,
                    rgb_path=rgb_path,
                    save_path=depth_path,
                    size=args.size,
                    watermark=args.watermark,
                    seed=args.seed,
                    timeout=args.timeout,
                    depth_polarity=args.depth_polarity,
                    analysis_text=analysis_text,
                    example_rgb=example_rgb,
                    example_depth=example_depth,
                )
                item["depth_mode"] = generation_result["mode"]
                item["image_url"] = generation_result["image_url"]
                item["request_id"] = generation_result["request_id"]
                item["relative_depth_output"] = depth_path.name
                output_meta = {
                    "image_name": rgb_path.name,
                    "relative_depth_output": depth_path.name,
                    "input_mode": args.input_mode,
                    "run_signature": run_signature,
                    "generation_result": generation_result,
                }
                if analysis_path is not None:
                    output_meta["analysis_name"] = analysis_path.name
                if args.input_mode == "rgb_plus_example":
                    output_meta["example_rgb"] = str(example_rgb) if example_rgb else ""
                    output_meta["example_depth"] = str(example_depth) if example_depth else ""
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
