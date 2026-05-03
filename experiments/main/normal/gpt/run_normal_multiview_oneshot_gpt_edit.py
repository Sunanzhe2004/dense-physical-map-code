#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
try:
    from openai import AzureOpenAI
except Exception as e:
    AzureOpenAI = None
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

DEFAULT_AZURE_ENDPOINT = "https://aif-icdevai02-eee-xjq-use2.cognitiveservices.azure.com/"
DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_ANALYSIS_MODEL = "gpt-4o-mini"
DEFAULT_NORMAL_MODEL = "gpt-image-1.5"
DEFAULT_NORMAL_SIZE = "1536x1024"
DEFAULT_NORMAL_QUALITY = "high"
DEFAULT_EDIT_CANVAS_SIZE = (1536, 1024)
DEFAULT_FINAL_OUTPUT_SIZE = (640, 480)
DEFAULT_EXAMPLE_RGB = "/path/to/benchmark_examples/normal/example1_rgb.png"
DEFAULT_EXAMPLE_NORMAL = "/path/to/benchmark_examples/normal/example1_normal.png"
DEFAULT_GT_ROOT = "/path/to/benchmark_data/GT"
DEFAULT_OUTPUT_DIR = "/path/to/benchmark_outputs/normal_gpt"
DEFAULT_EXTERNAL_META_DIRS = [
    "/path/to/benchmark_outputs/normal_doubao/interiorverse_mainaxis/meta",
    "/path/to/benchmark_outputs/normal_doubao/interiorverse_stresstest/meta",
    "/path/to/benchmark_outputs/normal_doubao/openroomff_mainaxis/meta",
    "/path/to/benchmark_outputs/normal_doubao/openroomff_stresstest/meta",
]
KNOWN_DATASET_NAMES = {
    "interiorverse_mainaxis",
    "interiorverse_stresstest",
    "openroomff_mainaxis",
    "openroomff_stresstest",
}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-view normal map generation (one-shot minimal v3k_colorsem_texturepatch_v4; stronger top-facing calibration, flat-top stability, shading-to-bump veto, and tighter fabric simplification)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--filename_suffix", type=str, default="_im.png", help="Only process filenames ending with this suffix.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input_dir.")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--azure_endpoint",
        "--base_url",
        dest="azure_endpoint",
        type=str,
        default=os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
        help="Azure OpenAI endpoint (shared fallback; role-specific endpoint has higher priority).",
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )
    parser.add_argument("--analysis_api_key", type=str, default=None)
    parser.add_argument(
        "--analysis_endpoint",
        "--analysis_azure_endpoint",
        dest="analysis_endpoint",
        type=str,
        default=os.environ.get("AZURE_ANALYSIS_OPENAI_ENDPOINT"),
    )
    parser.add_argument(
        "--analysis_api_version",
        type=str,
        default=os.environ.get("AZURE_ANALYSIS_OPENAI_API_VERSION"),
    )
    parser.add_argument(
        "--meta_dirs",
        type=str,
        nargs="*",
        default=DEFAULT_EXTERNAL_META_DIRS,
        help="Meta directories used to match samples. The script looks up records in manifest.json by relative_image_path.",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default=DEFAULT_GT_ROOT,
        help="Root directory of the original GT images. Legacy input_dir paths can be remapped through this root.",
    )
    parser.add_argument("--normal_api_key", type=str, default=None)
    parser.add_argument(
        "--normal_endpoint",
        "--normal_azure_endpoint",
        dest="normal_endpoint",
        type=str,
        default=os.environ.get("AZURE_NORMAL_OPENAI_ENDPOINT"),
    )
    parser.add_argument(
        "--normal_api_version",
        type=str,
        default=os.environ.get("AZURE_NORMAL_OPENAI_API_VERSION"),
    )
    parser.add_argument(
        "--analysis_model",
        type=str,
        default=DEFAULT_ANALYSIS_MODEL,
        help="Analysis model (used in zero-shot mode).",
    )
    parser.add_argument("--analysis_deployment", type=str, default=None)
    parser.add_argument("--normal_model", type=str, default=DEFAULT_NORMAL_MODEL, help="Normal model.")
    parser.add_argument("--normal_deployment", type=str, default=None, help="Normal image deployment.")
    parser.add_argument(
        "--max_views",
        type=int,
        default=6,
        help="Max views for analysis (zero-shot only).",
    )
    parser.add_argument(
        "--analysis_max_side",
        type=int,
        default=1024,
        help="Resize long side before analysis.",
    )
    parser.add_argument(
        "--detail",
        type=str,
        default="high",
        choices=["low", "high", "auto"],
        help="Vision detail level.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--normal_size", type=str, default=DEFAULT_NORMAL_SIZE, help="Normal output size.")
    parser.add_argument(
        "--normal_quality",
        type=str,
        default=DEFAULT_NORMAL_QUALITY,
        choices=["low", "medium", "high", "auto"],
        help="Normal image quality.",
    )
    parser.add_argument("--watermark", action="store_true", help="Keep watermark.")
    parser.add_argument("--max_generate", type=int, default=0, help="Max images to generate (0 = all).")
    parser.add_argument(
        "--shard_count",
        type=int,
        default=1,
        help="Total number of stable shards for this input directory.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index to process, in [0, shard_count).",
    )
    parser.add_argument(
        "--pending_start",
        type=int,
        default=0,
        help="Start offset within the pending image list, after completed outputs are filtered.",
    )
    parser.add_argument(
        "--pending_limit",
        type=int,
        default=0,
        help="Max pending images to process from --pending_start (0 = all remaining).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs and analysis files.",
    )
    parser.add_argument(
        "--independent_images",
        action="store_true",
        help="Treat each image as an independent sample and analyze/generate it separately.",
    )
    parser.add_argument(
        "--preserve_relative_dirs",
        action="store_true",
        help="Preserve relative directory layout under input_dir.",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests.")
    parser.add_argument("--timeout", type=int, default=120, help="Download timeout in seconds.")
    parser.add_argument(
        "--generation_retries",
        type=int,
        default=3,
        help="Retry count for transient generation API failures.",
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=5.0,
        help="Base sleep seconds between generation retries.",
    )
    parser.add_argument(
        "--example_rgb",
        type=str,
        default=DEFAULT_EXAMPLE_RGB,
        help="One-shot example RGB path. Default enables one-shot mode.",
    )
    parser.add_argument(
        "--example_normal",
        type=str,
        default=DEFAULT_EXAMPLE_NORMAL,
        help="One-shot example normal path. Default enables one-shot mode.",
    )
    parser.add_argument(
        "--save_debug_intermediates",
        action="store_true",
        help="Save full-canvas and cropped intermediate images for debugging padding/cropping drift.",
    )
    return parser.parse_args()


def resolve_azure_client_config(
    role: str,
    cli_api_key: Optional[str],
    cli_endpoint: Optional[str],
    cli_api_version: Optional[str],
    args: argparse.Namespace,
) -> Tuple[str, str, str]:
    role = role.lower().strip()
    if role not in {"analysis", "normal"}:
        raise ValueError(f"Unknown role: {role}")

    if role == "analysis":
        role_env_api_key = (
            os.environ.get("AZURE_ANALYSIS_OPENAI_API_KEY")
            or os.environ.get("AZURE_GPT4O_MINI_API_KEY")
        )
        role_env_endpoint = (
            os.environ.get("AZURE_ANALYSIS_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_GPT4O_MINI_ENDPOINT")
        )
        role_env_api_version = (
            os.environ.get("AZURE_ANALYSIS_OPENAI_API_VERSION")
            or os.environ.get("AZURE_GPT4O_MINI_API_VERSION")
        )
    else:
        role_env_api_key = (
            os.environ.get("AZURE_NORMAL_OPENAI_API_KEY")
            or os.environ.get("AZURE_GPT_IMAGE_15_API_KEY")
        )
        role_env_endpoint = (
            os.environ.get("AZURE_NORMAL_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT")
        )
        role_env_api_version = (
            os.environ.get("AZURE_NORMAL_OPENAI_API_VERSION")
            or os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION")
        )

    api_key = (
        cli_api_key
        or role_env_api_key
        or args.api_key
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            f"Missing {role} API key. Provide --{role}_api_key, --api_key, or the matching environment variable."
        )

    endpoint = cli_endpoint or role_env_endpoint or args.azure_endpoint or DEFAULT_AZURE_ENDPOINT
    api_version = cli_api_version or role_env_api_version or args.api_version or DEFAULT_API_VERSION
    return api_key, endpoint, api_version



def list_images(input_dir: Path, filename_suffix: Optional[str] = None, recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if filename_suffix:
        images = [p for p in images if p.name.endswith(filename_suffix)]
    if not images:
        suffix_msg = f" ending with {filename_suffix}" if filename_suffix else ""
        recursive_msg = " (recursive scan)" if recursive else " (top-level scan only)"
        raise FileNotFoundError(
            f"No supported image files found{suffix_msg}{recursive_msg}: {input_dir}"
        )
    return images


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


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    with Image.open(path) as img:
        return img.size


def _edge_fill_canvas(canvas: "Image.Image", resized: "Image.Image", left: int, top: int) -> None:
    rw, rh = resized.size
    cw, ch = canvas.size

    # left/right edge extension
    if left > 0:
        left_strip = resized.crop((0, 0, 1, rh)).resize((left, rh), Image.Resampling.BILINEAR)
        canvas.paste(left_strip, (0, top))
    right = left + rw
    if right < cw:
        right_strip = resized.crop((rw - 1, 0, rw, rh)).resize((cw - right, rh), Image.Resampling.BILINEAR)
        canvas.paste(right_strip, (right, top))

    # top/bottom extension after horizontal content is in place
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
    if Image is None:
        raise RuntimeError("This pipeline requires Pillow to run resize, padding, and cropping.")

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

        # Use edge-replicated padding instead of pure black padding.
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
        raise ValueError(f"bbox must contain 4 values. Current value: {bbox}")

    ref_w, ref_h = reference_size
    img_w, img_h = image_size
    if ref_w <= 0 or ref_h <= 0 or img_w <= 0 or img_h <= 0:
        raise ValueError(
            f"Invalid size: reference_size={reference_size}, image_size={image_size}"
        )

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
    if Image is None:
        raise RuntimeError("This pipeline requires Pillow to run resize, padding, and cropping.")

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        if crop_bbox is not None:
            img = img.crop(scale_bbox_to_image(crop_bbox, crop_reference_size, img.size))
        if final_size is not None:
            img = img.resize(final_size, Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()


def detect_dataset_name(path_like: Path) -> Optional[str]:
    parts = [part for part in path_like.parts if part not in {path_like.anchor, ""}]
    for part in reversed(parts):
        if part in KNOWN_DATASET_NAMES:
            return part
    if path_like.name in KNOWN_DATASET_NAMES:
        return path_like.name
    return None


def resolve_listing_input_dir(input_dir: Path, gt_root: Path) -> Path:
    if input_dir.exists():
        return input_dir
    dataset_name = detect_dataset_name(input_dir)
    if dataset_name:
        mapped_dir = gt_root / dataset_name
        if mapped_dir.exists():
            return mapped_dir
    raise FileNotFoundError(
        f"input_dir does not exist and no matching dataset was found under gt_root: input_dir={input_dir.as_posix()} gt_root={gt_root.as_posix()}"
    )


def candidate_relative_paths(image_path: Path, input_dir: Path, dataset_name: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def add_candidate(path_like: Path) -> None:
        key = path_like.as_posix().lstrip("/")
        if key and key not in seen:
            seen.add(key)
            candidates.append(path_like)

    try:
        add_candidate(image_path.relative_to(input_dir))
    except Exception:
        pass

    parts = [part for part in image_path.parts if part not in {image_path.anchor, ""}]
    if dataset_name and dataset_name in parts:
        last_idx = max(idx for idx, part in enumerate(parts) if part == dataset_name)
        if last_idx + 1 < len(parts):
            add_candidate(Path(*parts[last_idx + 1 :]))

    for start in range(len(parts)):
        add_candidate(Path(*parts[start:]))
    return candidates


def build_prompt_sources(meta_dirs: List[str]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for meta_dir_str in meta_dirs:
        meta_dir = Path(meta_dir_str)
        manifest_path = meta_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"[warn] Manifest does not exist; skipping: {manifest_path.as_posix()}", file=sys.stderr)
            continue
        manifest = load_manifest(manifest_path)
        prompt_map: Dict[str, Dict[str, Any]] = {}
        for item in manifest:
            relative_path = item.get("relative_image_path")
            if not relative_path:
                continue
            prompt_map[str(relative_path).lstrip("/")] = item
        dataset_name = meta_dir.parent.name if meta_dir.parent.name in KNOWN_DATASET_NAMES else None
        sources.append({
            "meta_dir": meta_dir,
            "manifest_path": manifest_path,
            "dataset_name": dataset_name,
            "prompt_map": prompt_map,
        })
    return sources


def resolve_prompt_entry(image_path: Path, input_dir: Path, prompt_sources: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Path]:
    image_parts = [part for part in image_path.parts if part not in {image_path.anchor, ""}]
    preferred_sources = [
        source for source in prompt_sources
        if source.get("dataset_name") and source["dataset_name"] in image_parts
    ]
    fallback_sources = [source for source in prompt_sources if source not in preferred_sources]
    for sources in (preferred_sources, fallback_sources):
        for source in sources:
            dataset_name = source.get("dataset_name")
            for relative_path in candidate_relative_paths(image_path, input_dir, dataset_name):
                key = relative_path.as_posix().lstrip("/")
                entry = source["prompt_map"].get(key)
                if entry:
                    return entry, source["manifest_path"]
    raise FileNotFoundError(f"Matched manifest record was not found: image={image_path.as_posix()}")


def resolve_gt_image_path(image_path: Path, input_dir: Path, gt_root: Path) -> Path:
    dataset_name = detect_dataset_name(image_path) or detect_dataset_name(input_dir)
    if dataset_name:
        dataset_root = gt_root / dataset_name
        for relative_path in candidate_relative_paths(image_path, input_dir, dataset_name):
            candidate = dataset_root / relative_path
            if candidate.exists():
                return candidate
    if image_path.exists():
        return image_path
    raise FileNotFoundError(
        f"Matched source image was not found: image={image_path.as_posix()} gt_root={gt_root.as_posix()}"
    )


def pil_resize_bytes(path: Path, max_side: int) -> Tuple[bytes, str]:
    if Image is None:
        data = path.read_bytes()
        return data, guess_mime(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / float(long_side)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


def file_to_data_uri(path: Path, max_side: Optional[int] = None) -> str:
    if max_side is not None:
        data, mime = pil_resize_bytes(path, max_side=max_side)
    else:
        data = path.read_bytes()
        mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def clean_json_text(text: str) -> str:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    return text


def robust_json_loads(text: str) -> Dict[str, Any]:
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json_file(path)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str = "relative_image_path") -> None:
    entry_key = entry.get(key_field)
    if entry_key is None:
        manifest.append(entry)
        return
    for idx, item in enumerate(manifest):
        if item.get(key_field) == entry_key:
            manifest[idx] = entry
            return
    manifest.append(entry)


def extract_error_code(exc: Exception) -> str:
    text = str(exc)
    patterns = [
        r"'code':\s*'([^']+)'",
        r'"code":\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return exc.__class__.__name__


def should_skip_image_error(exc: Exception) -> bool:
    text = str(exc)
    skip_patterns = [
        "InputTextSensitiveContentDetected",
        "InputImageSensitiveContentDetected",
        "SensitiveContentDetected",
        "sensitive information",
        "sensitive content",
    ]
    return any(pattern in text for pattern in skip_patterns)


def is_retryable_generation_error(exc: Exception) -> bool:
    error_name = type(exc).__name__
    if error_name in {
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "InternalServerError",
        "RateLimitError",
    }:
        return True
    text = str(exc).lower()
    retry_patterns = [
        "connection error",
        "connection refused",
        "connecttimeout",
        "request timed out",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(pattern in text for pattern in retry_patterns)


def build_skip_path(meta_dir: Path) -> Path:
    return meta_dir / "skipped_images.json"


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    stage: str,
    exc: Exception,
) -> None:
    entry = {
        "skip_key": f"{image_path.relative_to(input_dir).as_posix()}::{stage}",
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "stage": stage,
        "error_code": extract_error_code(exc),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "skipped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    upsert_manifest_entry(skipped_images, entry, key_field="skip_key")
    skipped_path.write_text(json.dumps(skipped_images, ensure_ascii=False, indent=2), encoding="utf-8")


def was_image_skipped(skipped_images: List[Dict[str, Any]], image_path: Path, input_dir: Path) -> Optional[Dict[str, Any]]:
    relative_path = image_path.relative_to(input_dir).as_posix()
    for item in skipped_images:
        if item.get("relative_image_path") == relative_path:
            return item
    return None


def extract_output_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return output_text

    if hasattr(resp, "output"):
        texts: List[str] = []
        for item in resp.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                txt = getattr(c, "text", None)
                if txt:
                    texts.append(txt)
        if texts:
            return "\n".join(texts)

    choices = getattr(resp, "choices", None)
    if choices:
        texts: List[str] = []
        for choice in choices:
            message = getattr(choice, "message", None)
            if not message:
                continue
            content = getattr(message, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                texts.append(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        texts.append(part["text"])
                    else:
                        txt = getattr(part, "text", None)
                        if txt:
                            texts.append(txt)
        if texts:
            return "\n".join(texts)

    return str(resp)


def fetch_url_bytes(url: str, timeout: int = 120) -> bytes:
    output = io.BytesIO()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    return output.getvalue()


def get_image_response_bytes(image_item: Any, timeout: int = 120) -> bytes:
    url = getattr(image_item, "url", None)
    if url:
        return fetch_url_bytes(url, timeout=timeout)
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)
    raise RuntimeError("Response has neither url nor b64_json; cannot save image.")


def save_image_response(
    image_item: Any,
    save_path: Path,
    timeout: int = 120,
    crop_bbox: Optional[List[int]] = None,
    crop_reference_size: Tuple[int, int] = DEFAULT_EDIT_CANVAS_SIZE,
    final_size: Optional[Tuple[int, int]] = None,
) -> None:
    url = getattr(image_item, "url", None)
    if url:
        image_bytes = fetch_url_bytes(url, timeout=timeout)
        if crop_bbox is not None or final_size is not None:
            image_bytes = postprocess_generated_image_bytes(
                image_bytes,
                crop_bbox=crop_bbox,
                crop_reference_size=crop_reference_size,
                final_size=final_size,
            )
        save_path.write_bytes(image_bytes)
        return
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        image_bytes = base64.b64decode(b64_json)
        if crop_bbox is not None or final_size is not None:
            image_bytes = postprocess_generated_image_bytes(
                image_bytes,
                crop_bbox=crop_bbox,
                crop_reference_size=crop_reference_size,
                final_size=final_size,
            )
        save_path.write_bytes(image_bytes)
        return
    raise RuntimeError("Response has neither url nor b64_json; cannot save image.")


def build_analysis_prompt(image_names: List[str]) -> str:
    names_text = "\n".join(f"- {name}" for name in image_names)
    return f"""
You will be shown multiple RGB views of the same indoor scene. Your task is not to generate an image. Instead, provide a very short geometric summary that can support zero-shot normal-map generation.

Requirements:
1. Focus only on large-scale geometry, not texture, shadows, highlights, or color.
2. If a detail looks more like material texture than true shape, ignore it.
3. Keep the summary concise and retain only stable structural information.
4. Output JSON only, with no extra explanation.

Input view filenames:
{names_text}

Return exactly this JSON structure:
{{
  "scene_summary": "one-sentence scene summary",
  "geometry_notes": ["2 to 6 large-scale geometry notes"],
  "per_view": [
    {{
      "image_name": "must exactly match an input filename",
      "normal_hint": "one short geometry note for this view"
    }}
  ]
}}
""".strip()

def analyze_multiview(
    client: Any,
    image_paths: List[Path],
    model: str,
    detail: str = "high",
    analysis_max_side: int = 1024,
) -> Dict[str, Any]:
    image_names = [p.name for p in image_paths]
    prompt = build_analysis_prompt(image_names)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        image_url = {"url": file_to_data_uri(path, max_side=analysis_max_side)}
        if detail in {"low", "high"}:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=16384,
        temperature=0.1,
    )
    text = extract_output_text(resp)
    data = robust_json_loads(text)
    data.setdefault("scene_summary", "")
    data.setdefault("geometry_notes", [])
    data.setdefault("per_view", [])
    return data


def get_per_view_hints(global_analysis: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in global_analysis.get("per_view", []):
        image_name = item.get("image_name")
        if image_name:
            result[image_name] = item.get("normal_hint", "")
    return result


def make_zero_shot_prompt(scene_summary: str, geometry_notes: List[str], per_view_hint: str) -> str:
    notes = "; ".join(geometry_notes) if geometry_notes else "N/A"
    return (
        "Generate a strict view-space per-pixel surface normal map for an indoor scene. "
        "This is a dense geometric normal map, not an artistic rendering and not an edge visualization. "
        "Use RGB to encode normal direction only. "
        "Do not preserve albedo color, shading, highlights, reflections, wood grain, fabric color, wall color, floor color, or lighting tint. "
        "Suppress material texture and high-frequency appearance detail. "
        "If a detail looks like texture rather than true geometry, remove it. "
        "Preserve only object boundaries, large-scale geometry, broad curvature, panel recesses, frame thickness, and real shape transitions. "
        "Large planar or gently curved regions should be smooth and nearly texture-free. "
        "Visible door faces, window frames, sills, recess walls, sofa surfaces, and table surfaces are valid surfaces, not holes. "
        "Do not invent fake grooves, texture relief, or neon contour lines. "
        "Output a clean, pixel-aligned indoor normal map.\n\n"
        f"Scene summary: {scene_summary or 'N/A'}\n"
        f"Geometry notes: {notes}\n"
        f"View-specific hint: {per_view_hint or 'N/A'}"
    )


def make_one_shot_prompt() -> str:
    return (
        "You are given three images in order: "
        "(1) the query indoor RGB image to convert, "
        "(2) an example indoor RGB image, "
        "(3) the target normal map for that example. "

        "Generate the normal map for image (1). "
        "Image (3) is the only convention reference and has the highest priority for normal-map encoding, "
        "but use it only for palette convention, direction-family convention, surface-role convention, deeper-color convention, boundary style, saturation range, and large-region consistency. "
        "Image (2) is only a geometry-to-convention analogy example. "
        "Neither image (2) nor image (3) may change the composition, crop, camera pose, or visible object inventory of image (1). "

        "Critical rules: "
        "color must encode surface orientation only, never material color, never albedo, never lighting tint. "
        "Do not preserve wood color, fabric color, wall color, floor color, or object appearance from the query RGB image. "
        "Do not preserve wood grain, fabric weave, printed patterns, shading, highlights, reflections, image noise, or any illumination residue. "

        "Treat image (1) only as a geometry-layout carrier. "
        "Preserve its exact pixel-space layout, exact current viewpoint, visible object inventory, crop extents, occlusion order, silhouettes, true boundaries, and visible geometry. "
        "Do not recrop, reframe, recompose, regularize, clean up, or canonicalize the query view. "
        "Do not simplify the scene to only broad shape. "
        "Overwrite photographic appearance with a clean synthetic normal-map encoding. "

        "Reference isolation rule: "
        "image (3) controls only the normal-map convention, palette family, sign convention, and orientation-family mapping. "
        "image (2) helps match analogous structural roles. "
        "Neither reference image controls scene composition, camera pose, object inventory, cropping, visibility, or which objects are present in image (1). "

        "Pixel-space fidelity rule: "
        "preserve the 2D image-space position, scale, and extent of all visible objects and boundaries. "
        "Do not zoom, shift, recenter, enlarge, shrink, or reposition visible objects. "
        "Keep the same crop, framing, and perspective footprint as the query image. "

        "Rigid layout preservation rule:"
        "Do not move, shift, recenter, enlarge, shrink, rotate, or re-place any visible object."
        "Every visible object must stay at the same 2D image-space position, with the same footprint, crop extent, and support contact, as in image (1)."
        "Do not recompose the scene."
        "Do not slide hanging objects, tabletop objects, wall-mounted objects, or foreground objects to a nearby easier position."
        "If uncertain, keep the original object position rather than improving composition."

        "Instance preservation rule: "
        "preserve all clearly visible object instances in image (1). "
        "Do not delete, omit, merge away, dissolve, or collapse visible furniture, foreground objects, tabletop objects, decor objects, wall-mounted objects, ceiling-mounted objects, or partially visible objects. "
        "If an object is clearly visible in the query image, it must remain present in the output normal map at the same location with a recognizable silhouette and approximate visible faces. "

        "Inventory and crop rule: "
        "preserve the visible object inventory, crop extents, border truncation behavior, and occlusion order of the query image. "
        "Do not clean up, declutter, beautify, substitute, or reduce the number of visible objects. "
        "Objects touching the image boundary must remain truncated in the same way as in the query image. "

        "Small-object rule: "
        "for small objects such as bottles, books, cups, phones, remotes, plant pots, trays, bowls, candles, and decor items, preserve the object instance itself, its count, location, silhouette, support contact, and main visible faces. "
        "Simplify tiny appearance details and texture, but do not delete the object, substitute it with a different object, merge multiple objects into one, or merge it into the supporting surface. "

        "Thin-structure retention rule: "
        "preserve thin but visible structures such as branches, stems, lamp arms, chandelier arms, narrow object edges, chair legs, table legs, and frame borders when they are clearly visible in image (1). "
        "Represent them with simplified but still visible geometry. "

        "Orientation separation rule: "
        "distinguish clearly between front-facing vertical faces, upward-facing horizontal faces, downward-facing undersides, and side-facing faces. "
        "Adjacent orthogonal faces should use distinct reference families when the reference normal map separates them. "
        "Wall, floor, ceiling, top surfaces, underside surfaces, and side-facing surfaces must remain separated by orientation, not by photographic shading. "

        "Global position-gradient veto rule: "
        "do not create a top-to-bottom, bottom-to-top, center-to-edge, or depth-like color drift across the whole image. "
        "Image position, distance, perspective depth, and scene brightness must not create a global purple-to-cyan or cyan-to-purple gradient. "

        "Lighting-removal rule: "
        "ignore illumination completely. "
        "Do not encode lamp glow, ceiling-light hotspots, window brightening, cast shadows, soft shadow gradients, interreflection, ambient occlusion-like darkening, exposure falloff, bloom, glare, or colored light spill into the normal map. "
        "Only true geometric orientation and the reference convention may determine output color. "

        "Planar family consistency rule: "
        "for one continuous planar region with nearly constant orientation, use one stable reference family across the whole region. "
        "Do not introduce large blue-purple or cyan-magenta drift across the same wall, same cabinet face, same desktop, same drawer front, same shelf face, same table top, or same curtain panel when the orientation is nearly constant. "

        "Dataset-style color restraint rule: "
        "prefer stable dataset-like normal colors over dramatic purple-cyan room-wide styling. "
        "Do not let ceiling, wall, and floor separation become a stylized global color design. "

        "Very strong texture suppression rule: "
        "remove fine-scale and mid-scale material texture. "
        "If a local variation is not clearly a true shape change, treat it as texture and suppress it. "
        "Large door faces, table tops, sofa backs, sofa seats, sofa arms, walls, floors, ceilings, and other broad surfaces should be smooth, low-variance, and nearly texture-free. "

        "Geometry rule: "
        "preserve the query image layout, exact current viewpoint, object inventory, object boundaries, major silhouettes, broad curvature, curtain folds, furniture macro-shape, panel recesses, frame thickness, contact boundaries, and true shape discontinuities. "
        "Use narrow transitions only at true geometric boundaries. "
        "Do not invent fake bevels, fake grooves, fake seams, texture relief, or artistic contour lines. "

        "Priority order: "
        "1) preserve the exact current query viewpoint, visible object inventory, crop extents, and layout, "
        "2) preserve all clearly visible object instances, their count, and occlusion order, "
        "3) preserve pixel-space positions and border truncation behavior, "
        "4) infer the meaning of the reference color families from image (3), "
        "5) keep wall, floor, ceiling, top surfaces, underside surfaces, and side-facing surfaces clearly separated when their orientations differ, "
        "6) keep one stable family for one nearly planar region, "
        "7) remove all lighting and illumination effects, "
        "8) suppress texture aggressively, "
        "9) never copy material color or material texture from image (1), "
        "10) if uncertain, prefer preserving a coarse visible object over deleting it. "

        "The output must look like a clean dataset normal map, not a purple relit image and not a stylized neon cyan-magenta rendering. "
        "Output only the final normal map for image (1)."
    )


def normalize_normal_size(size: Optional[str]) -> str:
    if not size:
        return DEFAULT_NORMAL_SIZE
    size = size.strip().lower()
    if size in {"adaptive", "source"}:
        return DEFAULT_NORMAL_SIZE
    return size


def run_image_generation(
    client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    quality: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
) -> Any:
    del watermark, seed, guidance_scale
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    size = normalize_normal_size(size)

    def _edit_with_fallback(image_arg: Any) -> Any:
        base_kwargs: Dict[str, Any] = {
            "model": model,
            "image": image_arg,
            "prompt": prompt,
            "n": 1,
        }
        fallback_variants: List[Dict[str, Any]] = [
            {"size": size, "quality": quality, "output_format": "png", "response_format": "b64_json"},
            {"size": size, "quality": quality, "output_format": "png"},
            {"size": size, "quality": quality},
            {"size": size},
            {},
        ]
        last_error: Optional[Exception] = None
        for extra_kwargs in fallback_variants:
            try:
                return client.images.edit(**base_kwargs, **extra_kwargs,input_fidelity="high")
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                if "Unknown parameter" in str(exc):
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("images.edit failed and no concrete fallback error was captured.")

    with contextlib.ExitStack() as stack:
        image_files = [stack.enter_context(path.open("rb")) for path in image_paths]
        image_arg: Any
        if len(image_files) == 1:
            image_arg = image_files[0]
        else:
            image_arg = image_files
        return _edit_with_fallback(image_arg)


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


def get_pending_images(
    image_paths: List[Path],
    input_dir: Path,
    normal_dir: Path,
    overwrite: bool,
    preserve_relative_dirs: bool,
) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending: List[Path] = []
    for image_path in image_paths:
        out_normal = build_image_output_path(
            image_path=image_path,
            input_dir=input_dir,
            base_dir=normal_dir,
            suffix="_normal.png",
            preserve_relative_dirs=preserve_relative_dirs,
        )
        if is_completed_output(out_normal):
            continue
        pending.append(image_path)
    return pending


def main() -> None:
    args = parse_args()
    normal_api_key, normal_endpoint, normal_api_version = resolve_azure_client_config(
        role="normal",
        cli_api_key=args.normal_api_key,
        cli_endpoint=args.normal_endpoint,
        cli_api_version=args.normal_api_version,
        args=args,
    )

    if AzureOpenAI is None:
        raise ImportError("Failed to import openai. Please install: pip install openai") from _OPENAI_IMPORT_ERROR

    example_rgb = Path(args.example_rgb) if args.example_rgb else None
    example_normal = Path(args.example_normal) if args.example_normal else None
    use_oneshot = bool(example_rgb and example_normal)

    if (example_rgb and not example_normal) or (example_normal and not example_rgb):
        raise ValueError("One-shot mode requires both --example_rgb and --example_normal.")

    if use_oneshot:
        if not example_rgb.exists():
            raise FileNotFoundError(f"example_rgb not found: {example_rgb}")
        if not example_normal.exists():
            raise FileNotFoundError(f"example_normal not found: {example_normal}")

    gt_root = Path(args.gt_root)
    requested_input_dir = Path(args.input_dir)
    input_dir = resolve_listing_input_dir(requested_input_dir, gt_root)
    output_dir = Path(args.output_dir)
    normal_dir = output_dir / "normal"
    meta_dir = output_dir / "meta"
    normal_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    if args.shard_count < 1:
        raise ValueError("--shard_count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard_index must be in [0, shard_count)")
    sharded_image_paths = image_paths_for_generate[args.shard_index :: args.shard_count]
    pending_image_paths = get_pending_images(
        image_paths=sharded_image_paths,
        input_dir=input_dir,
        normal_dir=normal_dir,
        overwrite=args.overwrite,
        preserve_relative_dirs=args.preserve_relative_dirs,
    )
    if args.pending_start < 0:
        raise ValueError("--pending_start must be >= 0")
    if args.pending_limit < 0:
        raise ValueError("--pending_limit must be >= 0")
    pending_end = None if args.pending_limit == 0 else args.pending_start + args.pending_limit
    selected_image_paths = pending_image_paths[args.pending_start : pending_end]

    one_shot_prompt = make_one_shot_prompt()

    print(
        f"[1/3] Found {len(image_paths)} images; {len(image_paths_for_generate)} candidate generations; "
        f"shard candidates: {len(sharded_image_paths)}; "
        f"pending: {len(pending_image_paths)}; processing in this run: {len(selected_image_paths)}; "
        f"shard={args.shard_index}/{args.shard_count}; "
        f"pending_start={args.pending_start} pending_limit={args.pending_limit}; one_shot={use_oneshot}"
    )
    print(
        "      "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"independent_images={args.independent_images} | "
        f"overwrite={args.overwrite} | "
        f"normal_model={args.normal_model} | "
        f"normal_deployment={args.normal_deployment or args.normal_model} | "
        f"normal_endpoint={normal_endpoint} | "
        f"normal_api_version={normal_api_version} | "
        f"normal_quality={args.normal_quality} | "
        f"normal_size={normalize_normal_size(args.normal_size)} | "
        f"gt_root={gt_root.as_posix()}"
    )

    normal_deployment = args.normal_deployment or args.normal_model
    normal_client = AzureOpenAI(
        api_version=normal_api_version,
        azure_endpoint=normal_endpoint,
        api_key=normal_api_key,
    )

    manifest_path = meta_dir / "manifest.json"
    skipped_path = build_skip_path(meta_dir)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)

    if requested_input_dir != input_dir:
        print(
            f"[info] Detected a migrated input_dir and remapped it automatically: "
            f"{requested_input_dir.as_posix()} -> {input_dir.as_posix()}"
        )

    meta_info = {
        "mode": "one_shot_query_first_edgepad_v1",
        "analysis_skipped": True,
        "prompt_source": "make_one_shot_prompt_query_first_edgepad",
        "example_rgb": str(example_rgb),
        "example_normal": str(example_normal),
    }
    (meta_dir / "multiview_analysis.json").write_text(
        json.dumps(meta_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[2/3] Upstream image analysis has been removed; the generation stage now uses the built-in make_one_shot_prompt directly.")
    print(f"[3/3] Start exporting normal maps ({len(selected_image_paths)} images)")

    if not pending_image_paths:
        print("  - The current directory is already complete; exiting.")
    elif not selected_image_paths:
        print("  - The current pending slice is empty; exiting.")

    for idx, image_path in enumerate(selected_image_paths, start=1):
        out_normal = build_image_output_path(
            image_path=image_path,
            input_dir=input_dir,
            base_dir=normal_dir,
            suffix="_normal.png",
            preserve_relative_dirs=args.preserve_relative_dirs,
        )
        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(
                f"  - ({idx}/{len(selected_image_paths)}) {image_path.name} "
                f"previously skipped, reason={skipped_entry.get('error_code', 'N/A')}"
            )
            continue
        if not args.overwrite and is_completed_output(out_normal):
            print(f"  - ({idx}/{len(selected_image_paths)}) {image_path.name} already exists; skipping")
            continue
        print(f"  - ({idx}/{len(selected_image_paths)}) {image_path.name}")

        try:
            normal_prompt = one_shot_prompt
            source_image_path = resolve_gt_image_path(image_path, input_dir, gt_root)
        except Exception as exc:
            print(f"      Skip: failed to read the GT source image -> {exc}")
            record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "gt_lookup", exc)
            continue

        try:
            with tempfile.TemporaryDirectory(prefix="gpt_edit_pad_") as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                padded_example_rgb = temp_dir / "example_rgb.png"
                padded_example_normal = temp_dir / "example_normal.png"
                padded_query = temp_dir / "query.png"

                example_rgb_bbox = preprocess_image_with_padding(example_rgb, padded_example_rgb)
                example_normal_bbox = preprocess_image_with_padding(example_normal, padded_example_normal)
                query_bbox = preprocess_image_with_padding(source_image_path, padded_query)
                normal_input_paths = [padded_query, padded_example_rgb, padded_example_normal]

                max_attempts = max(1, args.generation_retries + 1)
                for attempt in range(1, max_attempts + 1):
                    try:
                        normal_resp = run_image_generation(
                            client=normal_client,
                            model=normal_deployment,
                            prompt=normal_prompt,
                            image_paths=normal_input_paths,
                            size=args.normal_size,
                            quality=args.normal_quality,
                            watermark=args.watermark,
                            seed=args.seed,
                            guidance_scale=args.guidance_scale,
                        )
                        break
                    except Exception as exc:
                        if attempt < max_attempts and is_retryable_generation_error(exc):
                            wait_seconds = max(0.0, args.retry_sleep) * attempt
                            print(
                                f"      Retry: transient generation error {type(exc).__name__}, "
                                f"{wait_seconds:.1f}s before retry {attempt}/{max_attempts - 1}"
                            )
                            time.sleep(wait_seconds)
                            continue
                        raise
            ensure_parent_dir(out_normal)
            raw_bytes = get_image_response_bytes(normal_resp.data[0], timeout=args.timeout)

            if args.save_debug_intermediates:
                debug_full = build_image_output_path(
                    image_path=image_path,
                    input_dir=input_dir,
                    base_dir=normal_dir,
                    suffix="_normal_full.png",
                    preserve_relative_dirs=args.preserve_relative_dirs,
                )
                debug_crop = build_image_output_path(
                    image_path=image_path,
                    input_dir=input_dir,
                    base_dir=normal_dir,
                    suffix="_normal_crop.png",
                    preserve_relative_dirs=args.preserve_relative_dirs,
                )
                ensure_parent_dir(debug_full)
                debug_full.write_bytes(raw_bytes)
                debug_crop.write_bytes(
                    postprocess_generated_image_bytes(
                        raw_bytes,
                        crop_bbox=query_bbox,
                        crop_reference_size=DEFAULT_EDIT_CANVAS_SIZE,
                        final_size=None,
                    )
                )

            out_normal.write_bytes(
                postprocess_generated_image_bytes(
                    raw_bytes,
                    crop_bbox=query_bbox,
                    crop_reference_size=DEFAULT_EDIT_CANVAS_SIZE,
                    final_size=DEFAULT_FINAL_OUTPUT_SIZE,
                )
            )
        except Exception as exc:
            if should_skip_image_error(exc):
                print(f"      Skip: generation stage triggered a skippable error -> {extract_error_code(exc)}")
                record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc)
                continue
            raise

        upsert_manifest_entry(
            manifest,
            {
                "image_name": image_path.name,
                "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                "analysis_model": "external_manifest_json",
                "analysis_deployment": "external_manifest_json",
                "analysis_endpoint": "",
                "analysis_api_version": "",
                "normal_model": args.normal_model,
                "normal_deployment": normal_deployment,
                "normal_endpoint": normal_endpoint,
                "normal_api_version": normal_api_version,
                "normal_quality": args.normal_quality,
                "normal_size": normalize_normal_size(args.normal_size),
                "preprocess_canvas_size": f"{DEFAULT_EDIT_CANVAS_SIZE[0]}x{DEFAULT_EDIT_CANVAS_SIZE[1]}",
                "saved_normal_size": f"{DEFAULT_FINAL_OUTPUT_SIZE[0]}x{DEFAULT_FINAL_OUTPUT_SIZE[1]}",
                "mode": "one_shot_v3k_colorsem_texturepatch_v4",
                "example_rgb": str(example_rgb) if example_rgb else "",
                "example_normal": str(example_normal) if example_normal else "",
                "example_rgb_content_bbox": example_rgb_bbox,
                "example_normal_content_bbox": example_normal_bbox,
                "query_content_bbox": query_bbox,
                "source_image_path": source_image_path.as_posix(),
                "normal_output": out_normal.relative_to(output_dir).as_posix(),
                "analysis_output": "",
                "normal_prompt": normal_prompt,
                "input_order": ["query_rgb", "example_rgb", "example_normal"],
                "padding_mode": "edge_replicate",
                "save_debug_intermediates": bool(args.save_debug_intermediates),
            },
        )
        time.sleep(max(0.0, args.sleep))

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Normal output dir: {normal_dir.as_posix()}")
    print(f"Meta output dir: {meta_dir.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nInterrupted.", file=sys.stderr)
        sys.exit(130)
