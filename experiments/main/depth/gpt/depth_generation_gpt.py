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
import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

try:
    from openai import AzureOpenAI, OpenAI
except Exception as e:
    AzureOpenAI = None
    OpenAI = None
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None


DEFAULT_AZURE_ENDPOINT = "https://aif-icdevai02-eee-xjq-use2.cognitiveservices.azure.com/"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_BASE_URL = DEFAULT_AZURE_ENDPOINT
DEFAULT_IMAGE_MODEL = "gpt-image-1.5"
DEFAULT_IMAGE_QUALITY = "medium"
DEFAULT_IMAGE_SIZE = "1536x1024"
DEFAULT_EDIT_CANVAS_SIZE = (1536, 1024)
DEFAULT_TIMEOUT = 1800

PROMPT_VERSION = "relative_depth_benchmark_v9_near_white_only"
DEPTH_POLARITY_CHOICES = ("near_white",)
FIXED_DEPTH_POLARITY = "near_white"

INPUT_MODE_CHOICES = ("rgb_only", "rgb_plus_analysis", "rgb_plus_example")
SUPPORTED_GPT_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate relative depth maps with GPT Image 1.5 from RGB inputs."
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
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI or Azure OpenAI API key")
    parser.add_argument(
        "--base_url",
        "--azure_endpoint",
        dest="base_url",
        type=str,
        default=os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or DEFAULT_BASE_URL,
        help="Azure OpenAI endpoint or OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or DEFAULT_API_VERSION,
        help="Azure OpenAI API version",
    )
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument("--size", type=str, default=DEFAULT_IMAGE_SIZE, help="GPT Image output size")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist")
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = (
        cli_api_key
        or os.environ.get("AZURE_GPT_IMAGE_15_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if api_key:
        api_key = api_key.strip()
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                "API key contains non-ASCII characters and cannot be written to the HTTP header. "
                "Please replace any placeholder text with the real key."
            ) from e
        return api_key
    raise RuntimeError(
        "Missing authentication: please provide --api_key or set "
        "AZURE_GPT_IMAGE_15_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY."
    )


def resolve_image_client_config(args: argparse.Namespace) -> Any:
    api_key = ensure_api_key(args.api_key)
    endpoint = args.base_url or os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = args.api_version or os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION") or os.environ.get("AZURE_OPENAI_API_VERSION")
    if endpoint:
        if AzureOpenAI is None:
            raise ImportError("Failed to import openai. Please install: pip install openai") from _OPENAI_IMPORT_ERROR
        return AzureOpenAI(
            api_version=api_version or DEFAULT_API_VERSION,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
    if OpenAI is None:
        raise ImportError("Failed to import openai. Please install: pip install openai") from _OPENAI_IMPORT_ERROR
    return OpenAI(api_key=api_key)


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


def normalize_gpt_image_size(size: Optional[str]) -> str:
    normalized = (size or DEFAULT_IMAGE_SIZE).strip().lower()
    if normalized in {"adaptive", "source", "2k"}:
        normalized = DEFAULT_IMAGE_SIZE
    if normalized not in SUPPORTED_GPT_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported --size for gpt-image-1.5: {size}. "
            f"Choices: {', '.join(sorted(SUPPORTED_GPT_IMAGE_SIZES))}"
        )
    return normalized


def _edge_fill_canvas(canvas: "Image.Image", resized: "Image.Image", left: int, top: int) -> None:
    rw, rh = resized.size
    cw, ch = canvas.size

    if left > 0:
        left_strip = resized.crop((0, 0, 1, rh)).resize((left, rh), Image.Resampling.BILINEAR)
        canvas.paste(left_strip, (0, top))
    right = left + rw
    if right < cw:
        right_strip = resized.crop((rw - 1, 0, rw, rh)).resize((cw - right, rh), Image.Resampling.BILINEAR)
        canvas.paste(right_strip, (right, top))

    if top > 0:
        top_band = canvas.crop((0, top, cw, top + 1)).resize((cw, top), Image.Resampling.BILINEAR)
        canvas.paste(top_band, (0, 0))
    bottom = top + rh
    if bottom < ch:
        bottom_band = canvas.crop((0, bottom - 1, cw, bottom)).resize((cw, ch - bottom), Image.Resampling.BILINEAR)
        canvas.paste(bottom_band, (0, bottom))


def preprocess_image_with_padding(
    src_path: Path,
    dst_path: Path,
    target_size: Tuple[int, int] = DEFAULT_EDIT_CANVAS_SIZE,
) -> List[int]:
    target_w, target_h = target_size
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        src_w, src_h = img.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"Invalid image size: {src_path.as_posix()} -> {(src_w, src_h)}")

        scale = min(target_w / src_w, target_h / src_h)
        resized_w = max(1, min(target_w, int(round(src_w * scale))))
        resized_h = max(1, min(target_h, int(round(src_h * scale))))
        resized = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

        left = (target_w - resized_w) // 2
        top = (target_h - resized_h) // 2
        canvas = Image.new("RGB", (target_w, target_h))
        canvas.paste(resized, (left, top))
        _edge_fill_canvas(canvas, resized, left, top)
        canvas.save(dst_path, format="PNG")
        return [left, top, left + resized_w, top + resized_h]


def scale_bbox_to_image(
    bbox: List[int],
    reference_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values: {bbox}")

    ref_w, ref_h = reference_size
    img_w, img_h = image_size
    if ref_w <= 0 or ref_h <= 0 or img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid sizes: reference_size={reference_size}, image_size={image_size}")

    scale_x = img_w / ref_w
    scale_y = img_h / ref_h
    left, top, right, bottom = bbox
    scaled_left = int(round(left * scale_x))
    scaled_top = int(round(top * scale_y))
    scaled_right = int(round(right * scale_x))
    scaled_bottom = int(round(bottom * scale_y))

    scaled_left = max(0, min(scaled_left, img_w - 1))
    scaled_top = max(0, min(scaled_top, img_h - 1))
    scaled_right = max(scaled_left + 1, min(scaled_right, img_w))
    scaled_bottom = max(scaled_top + 1, min(scaled_bottom, img_h))
    return scaled_left, scaled_top, scaled_right, scaled_bottom


def postprocess_generated_image_bytes(
    image_bytes: bytes,
    crop_bbox: Optional[List[int]] = None,
    crop_reference_size: Tuple[int, int] = DEFAULT_EDIT_CANVAS_SIZE,
    final_size: Optional[Tuple[int, int]] = None,
) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        if crop_bbox is not None:
            img = img.crop(scale_bbox_to_image(crop_bbox, crop_reference_size, img.size))
        if final_size is not None:
            img = img.resize(final_size, Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()


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


def run_image_generation(
    image_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
) -> Any:
    del watermark
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    requested_size = normalize_gpt_image_size(size)

    with contextlib.ExitStack() as stack:
        image_files = [stack.enter_context(path.open("rb")) for path in image_paths]
        image_arg: Any = image_files[0] if len(image_files) == 1 else image_files
        return image_client.images.edit(
            model=model,
            image=image_arg,
            prompt=prompt,
            n=1,
            size=requested_size,
            quality=DEFAULT_IMAGE_QUALITY,
            output_format="png",
        )


def fetch_url_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    output = io.BytesIO()
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    return output.getvalue()


def get_image_response_bytes(image_item: Any, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    url = getattr(image_item, "url", None)
    if url:
        return fetch_url_bytes(url, timeout=timeout)

    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)

    if isinstance(image_item, dict):
        if image_item.get("url"):
            return fetch_url_bytes(image_item["url"], timeout=timeout)
        if image_item.get("b64_json"):
            return base64.b64decode(image_item["b64_json"])

    raise RuntimeError("Image response contains neither url nor b64_json.")


def save_image_response(
    image_item: Any,
    save_path: Path,
    timeout: int = DEFAULT_TIMEOUT,
    crop_bbox: Optional[List[int]] = None,
    crop_reference_size: Tuple[int, int] = DEFAULT_EDIT_CANVAS_SIZE,
    final_size: Optional[Tuple[int, int]] = None,
) -> None:
    image_bytes = get_image_response_bytes(image_item, timeout=timeout)
    if crop_bbox is not None or final_size is not None:
        image_bytes = postprocess_generated_image_bytes(
            image_bytes,
            crop_bbox=crop_bbox,
            crop_reference_size=crop_reference_size,
            final_size=final_size,
        )
    save_path.write_bytes(image_bytes)


def postprocess_depth_png(path: Path, depth_polarity: str) -> None:
    # Near-white only; no inversion is ever applied.
    del depth_polarity
    with Image.open(path) as image:
        image = image.convert("L")
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
    image_model: str,
    prompt_text: str,
    depth_polarity: str,
    analysis_dir: str,
    analysis_suffix: str,
    example_rgb: str,
    example_depth: str,
    timeout: int,
    size: str,
    watermark: bool,
) -> Dict[str, Any]:
    if input_mode == "rgb_plus_analysis":
        route = "gpt_rgb_relative_depth_analysis"
    elif input_mode == "rgb_plus_example":
        route = "gpt_rgb_relative_depth_example"
    else:
        route = "gpt_rgb_relative_depth_direct"
    return {
        "image_model": image_model,
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
        "size": size,
        "preprocess_canvas_size": f"{DEFAULT_EDIT_CANVAS_SIZE[0]}x{DEFAULT_EDIT_CANVAS_SIZE[1]}",
        "padding_mode": "edge_replicate",
        "output_resolution_policy": "crop_padding_and_restore_source_size",
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


def generate_relative_depth_map_with_gpt(
    *,
    image_client: Any,
    model: str,
    input_mode: str,
    rgb_path: Path,
    save_path: Path,
    size: str,
    watermark: bool,
    timeout: int,
    depth_polarity: str,
    analysis_text: str,
    example_rgb: Optional[Path],
    example_depth: Optional[Path],
) -> str:
    prompt = build_relative_depth_prompt(
        depth_polarity,
        analysis_text=analysis_text,
        input_mode=input_mode,
    )
    if input_mode == "rgb_plus_example":
        if not example_rgb or not example_depth:
            raise ValueError("rgb_plus_example requires both --example_rgb and --example_depth")
        mode = "gpt_rgb_relative_depth_example"
    else:
        mode = "gpt_rgb_relative_depth_analysis" if analysis_text else "gpt_rgb_relative_depth_direct"

    source_size = get_image_size(rgb_path)
    with tempfile.TemporaryDirectory(prefix="gpt_edit_pad_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        padded_rgb = temp_dir / "query.png"
        query_bbox = preprocess_image_with_padding(rgb_path, padded_rgb)

        if input_mode == "rgb_plus_example":
            if not example_rgb or not example_depth:
                raise ValueError("rgb_plus_example requires both --example_rgb and --example_depth")
            padded_example_rgb = temp_dir / "example_rgb.png"
            padded_example_depth = temp_dir / "example_depth.png"
            preprocess_image_with_padding(example_rgb, padded_example_rgb)
            preprocess_image_with_padding(example_depth, padded_example_depth)
            image_paths = [padded_example_rgb, padded_example_depth, padded_rgb]
        else:
            image_paths = [padded_rgb]

        response = run_image_generation(
            image_client=image_client,
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            size=size,
            watermark=watermark,
        )

    save_image_response(
        response.data[0],
        save_path,
        timeout=timeout,
        crop_bbox=query_bbox,
        crop_reference_size=DEFAULT_EDIT_CANVAS_SIZE,
        final_size=source_size,
    )
    postprocess_depth_png(save_path, depth_polarity=depth_polarity)
    return mode


def main() -> None:
    args = parse_args()

    if args.depth_polarity != FIXED_DEPTH_POLARITY:
        raise ValueError(
            f"This script only supports {FIXED_DEPTH_POLARITY}, got: {args.depth_polarity}"
        )

    image_client = resolve_image_client_config(args)

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
        image_model=args.image_model,
        prompt_text=prompt_text,
        depth_polarity=args.depth_polarity,
        analysis_dir=str(analysis_dir) if analysis_dir else "",
        analysis_suffix=args.analysis_suffix,
        example_rgb=str(example_rgb) if example_rgb else "",
        example_depth=str(example_depth) if example_depth else "",
        timeout=args.timeout,
        size=args.size,
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
                item["depth_mode"] = generate_relative_depth_map_with_gpt(
                    image_client=image_client,
                    model=args.image_model,
                    input_mode=args.input_mode,
                    rgb_path=rgb_path,
                    save_path=depth_path,
                    size=args.size,
                    watermark=args.watermark,
                    timeout=args.timeout,
                    depth_polarity=args.depth_polarity,
                    analysis_text=analysis_text,
                    example_rgb=example_rgb,
                    example_depth=example_depth,
                )
                item["relative_depth_output"] = depth_path.name
                output_meta = {
                    "image_name": rgb_path.name,
                    "relative_depth_output": depth_path.name,
                    "input_mode": args.input_mode,
                    "run_signature": run_signature,
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
