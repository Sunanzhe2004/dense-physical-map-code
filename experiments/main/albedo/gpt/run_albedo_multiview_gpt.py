#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

try:
    from openai import AzureOpenAI, OpenAI
except Exception as e:
    AzureOpenAI = None
    OpenAI = None
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from PIL import Image
except Exception:
    Image = None

DEFAULT_AZURE_ENDPOINT = "https://aif-icdevai02-eee-xjq-use2.cognitiveservices.azure.com/"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_ALBEDO_MODEL = "gpt-image-1.5"
DEFAULT_ALBEDO_QUALITY = "medium"
DEFAULT_GENERATION_MODE = "generate"
DEFAULT_GT_ROOT = "/path/to/benchmark_data/GT"
DEFAULT_EXTERNAL_ANALYSIS_DIRS = [
    "/path/to/benchmark_outputs/albedo_doubao/interiorverse_mainaxis/meta/per_image_analysis",
    "/path/to/benchmark_outputs/albedo_doubao/interiorverse_stresstest/meta/per_image_analysis",
    "/path/to/benchmark_outputs/albedo_doubao/openroomff_mainaxis/meta/per_image_analysis",
    "/path/to/benchmark_outputs/albedo_doubao/openroomff_stresstest/meta/per_image_analysis",
]
KNOWN_DATASET_NAMES = {
    "interiorverse_mainaxis",
    "interiorverse_stresstest",
    "openroomff_mainaxis",
    "openroomff_stresstest",
}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_GPT_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate albedo maps for multi-view indoor scenes.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_suffix", type=str, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan all subdirectories under input_dir.")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--azure_endpoint", "--base_url", dest="azure_endpoint", type=str, default=os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT))
    parser.add_argument("--api_version", type=str, default=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION))
    parser.add_argument("--analysis_api_key", type=str, default=None, help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--analysis_endpoint", "--analysis_azure_endpoint", dest="analysis_endpoint", type=str, default=os.environ.get("AZURE_ANALYSIS_OPENAI_ENDPOINT"), help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--analysis_api_version", type=str, default=os.environ.get("AZURE_ANALYSIS_OPENAI_API_VERSION"), help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--albedo_api_key", type=str, default=None)
    parser.add_argument("--albedo_endpoint", "--albedo_azure_endpoint", dest="albedo_endpoint", type=str, default=os.environ.get("AZURE_ALBEDO_OPENAI_ENDPOINT"))
    parser.add_argument("--albedo_api_version", type=str, default=os.environ.get("AZURE_ALBEDO_OPENAI_API_VERSION"))
    parser.add_argument("--analysis_model", type=str, default="external_json", help="Legacy compatibility argument.")
    parser.add_argument("--analysis_deployment", type=str, default=None, help="Legacy compatibility argument.")
    parser.add_argument("--albedo_model", type=str, default=DEFAULT_ALBEDO_MODEL)
    parser.add_argument("--albedo_deployment", type=str, default=None)
    parser.add_argument("--generation_mode", type=str, default=DEFAULT_GENERATION_MODE, choices=["generate", "edit"], help="Choose the generate endpoint or the image-edit endpoint.")
    parser.add_argument("--generate_requires_image", action="store_true", help="Require a valid image argument in generate mode instead of falling back to text-only generation.")
    parser.add_argument("--analysis_dirs", type=str, nargs="*", default=DEFAULT_EXTERNAL_ANALYSIS_DIRS, help="Directories containing precomputed per-image analysis JSON files.")
    parser.add_argument("--gt_root", type=str, default=DEFAULT_GT_ROOT, help="Root directory of the original GT images. Legacy input_dir paths can be remapped through this root.")
    parser.add_argument("--max_views", type=int, default=1, help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--analysis_max_side", type=int, default=1024, help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--detail", type=str, default="high", choices=["low", "high", "auto"], help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5, help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--albedo_size", type=str, default="1536x1024", help="Output size for gpt-image-1.5. Supported values: 1024x1024, 1024x1536, 1536x1024, auto.")
    parser.add_argument("--albedo_quality", type=str, default=DEFAULT_ALBEDO_QUALITY, choices=["low", "medium", "high", "auto"])
    parser.add_argument("--watermark", action="store_true", help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs instead of skipping completed results.")
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)
    parser.add_argument("--independent_images", action="store_true", help="Legacy compatibility argument. The current script always loads existing per-image analysis and processes images independently.")
    parser.add_argument("--preserve_relative_dirs", action="store_true", help="Preserve the relative directory layout from input_dir in the output tree.")
    parser.add_argument("--analysis_scope", type=str, default="part", choices=["full", "part"], help="Legacy compatibility argument; currently unused.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--request_retries", type=int, default=5, help="Retry count for network jitter, timeouts, rate limits, or 5xx errors during generation.")
    parser.add_argument("--retry_backoff", type=float, default=5.0, help="Initial backoff in seconds before the first generation retry.")
    parser.add_argument("--retry_max_backoff", type=float, default=60.0, help="Maximum wait time for a single generation retry.")
    return parser.parse_args()

def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = (
        cli_api_key
        or os.environ.get("AZURE_ALBEDO_OPENAI_API_KEY")
        or os.environ.get("AZURE_GPT_IMAGE_15_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing albedo API key. Provide --albedo_api_key, --api_key, or one of the environment variables "
            "AZURE_ALBEDO_OPENAI_API_KEY / AZURE_GPT_IMAGE_15_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY."
        )
    return api_key


def resolve_image_client_config(args: argparse.Namespace) -> Tuple[Any, str, str, Optional[str], Optional[str]]:
    api_key = ensure_api_key(args.albedo_api_key or args.api_key)
    endpoint = (
        args.albedo_endpoint
        or args.azure_endpoint
        or os.environ.get("AZURE_ALBEDO_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    api_version = (
        args.albedo_api_version
        or args.api_version
        or os.environ.get("AZURE_ALBEDO_OPENAI_API_VERSION")
        or os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
    )
    deployment = args.albedo_deployment or args.albedo_model
    is_azure = bool(args.albedo_endpoint or args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"))
    if is_azure:
        if AzureOpenAI is None:
            raise ImportError('Failed to import openai. Please install: pip install openai') from _OPENAI_IMPORT_ERROR
        client = AzureOpenAI(
            api_version=api_version or DEFAULT_API_VERSION,
            azure_endpoint=endpoint or DEFAULT_AZURE_ENDPOINT,
            api_key=api_key,
        )
        return client, deployment, "azure", endpoint, api_version or DEFAULT_API_VERSION
    if OpenAI is None:
        raise ImportError('Failed to import openai. Please install: pip install openai') from _OPENAI_IMPORT_ERROR
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if endpoint:
        client_kwargs["base_url"] = endpoint
    client = OpenAI(**client_kwargs)
    return client, deployment, "openai", endpoint, api_version


def validate_parts(num_parts: int, part_index: int) -> None:
    if num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if part_index < 0 or part_index >= num_parts:
        raise ValueError("--part_index must satisfy 0 <= part_index < num_parts")


def shard_paths(paths: List[Path], num_parts: int, part_index: int) -> List[Path]:
    if num_parts == 1:
        return list(paths)
    return [path for idx, path in enumerate(paths) if idx % num_parts == part_index]


def list_images(input_dir: Path, filename_suffix: Optional[str] = None, recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if filename_suffix:
        images = [p for p in images if p.name.endswith(filename_suffix)]
    if not images:
        suffix_msg = f" ending with {filename_suffix}" if filename_suffix else ""
        recursive_msg = " (recursive scan)" if recursive else " (top-level scan only)"
        raise FileNotFoundError(f"No supported image files found{suffix_msg}{recursive_msg}: {input_dir}")
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


def file_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    with Image.open(path) as img:
        return img.size


def normalize_gpt_image_size(size: Optional[str]) -> str:
    normalized = (size or "auto").strip().lower()
    if normalized not in SUPPORTED_GPT_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported --albedo_size: {size}. Supported values: {', '.join(sorted(SUPPORTED_GPT_IMAGE_SIZES))}"
        )
    return normalized


def normalize_analysis_data(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid analysis file format: expected a JSON object, got {type(data).__name__}")
    normalized = dict(data)
    normalized.setdefault("scene_summary", "")
    normalized.setdefault("albedo_material_notes", [])
    normalized.setdefault("albedo_lighting_notes", [])
    normalized.setdefault("albedo_prompt_suffix", "")
    normalized.setdefault("per_view", [])
    if not isinstance(normalized["albedo_material_notes"], list):
        normalized["albedo_material_notes"] = [str(normalized["albedo_material_notes"])]
    if not isinstance(normalized["albedo_lighting_notes"], list):
        normalized["albedo_lighting_notes"] = [str(normalized["albedo_lighting_notes"])]
    if not isinstance(normalized["per_view"], list):
        normalized["per_view"] = []
    return normalized


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


def save_manifest(path: Path, manifest: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str = "image_name") -> None:
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
        "moderation_blocked",
        "rejected by the safety system",
        "image_generation_user_error",
    ]
    return any(pattern in text for pattern in skip_patterns)


def should_retry_image_error(exc: Exception) -> bool:
    text = str(exc)
    retry_patterns = [
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "ConnectError",
        "ReadTimeout",
        "timed out",
        "Connection refused",
        "Connection reset",
        "connection aborted",
        "temporarily unavailable",
        "rate limit",
        "rate_limit",
        "Too Many Requests",
        '"status_code": 429',
        "'status_code': 429",
        '"status_code": 500',
        "'status_code': 500",
        '"status_code": 502',
        "'status_code': 502",
        '"status_code": 503',
        "'status_code': 503",
        '"status_code": 504',
        "'status_code': 504",
    ]
    return any(pattern in text for pattern in retry_patterns)


def build_skip_path(meta_dir: Path, num_parts: int, part_index: int) -> Path:
    filename = "skipped_images.json"
    if num_parts > 1:
        filename = f"skipped_images.part{part_index + 1}of{num_parts}.json"
    return meta_dir / filename


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    stage: str,
    exc: Exception,
    args: argparse.Namespace,
) -> None:
    entry = {
        "skip_key": f"{image_path.relative_to(input_dir).as_posix()}::{stage}",
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "stage": stage,
        "error_code": extract_error_code(exc),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "part_index": args.part_index,
        "num_parts": args.num_parts,
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


def save_url_to_file(url: str, save_path: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int = 120) -> None:
    url = getattr(image_item, "url", None)
    if url:
        save_url_to_file(url, save_path, timeout=timeout)
        return
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        save_path.write_bytes(base64.b64decode(b64_json))
        return
    raise RuntimeError("Response has neither url nor b64_json; cannot save the image.")


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


def build_analysis_sources(analysis_dirs: List[str]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for analysis_dir_str in analysis_dirs:
        analysis_dir = Path(analysis_dir_str)
        if not analysis_dir.exists():
            print(f"[warn] Analysis directory does not exist; skipping: {analysis_dir.as_posix()}", file=sys.stderr)
            continue
        dataset_name = analysis_dir.parent.parent.name if len(analysis_dir.parents) >= 2 else analysis_dir.name
        sources.append({
            "root": analysis_dir,
            "dataset_name": dataset_name,
        })
    return sources


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


def resolve_external_analysis_path(image_path: Path, input_dir: Path, analysis_sources: List[Dict[str, Any]]) -> Path:
    image_parts = [part for part in image_path.parts if part not in {image_path.anchor, ""}]
    preferred_sources = [
        source for source in analysis_sources
        if source.get("dataset_name") and source["dataset_name"] in image_parts
    ]
    fallback_sources = [source for source in analysis_sources if source not in preferred_sources]
    for sources in (preferred_sources, fallback_sources):
        for source in sources:
            dataset_name = source.get("dataset_name")
            for relative_path in candidate_relative_paths(image_path, input_dir, dataset_name):
                candidate = source["root"] / relative_path.with_suffix(".json")
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"Matched analysis JSON was not found: image={image_path.as_posix()}")


def load_external_analysis(image_path: Path, input_dir: Path, analysis_sources: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Path]:
    analysis_path = resolve_external_analysis_path(image_path, input_dir, analysis_sources)
    return normalize_analysis_data(load_json_file(analysis_path)), analysis_path


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


def get_per_view_hints(global_analysis: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    result = {}
    for item in global_analysis.get("per_view", []):
        image_name = item.get("image_name")
        if not image_name:
            continue
        result[image_name] = {"albedo_hint": item.get("albedo_hint", "")}
    return result


def make_albedo_prompt(global_analysis: Dict[str, Any], per_view_hint: str) -> str:
    material_notes = "; ".join(global_analysis.get("albedo_material_notes", []))
    lighting_notes = "; ".join(global_analysis.get("albedo_lighting_notes", []))
    suffix = global_analysis.get("albedo_prompt_suffix", "")
    base = (
        "Convert the input image into a clean intrinsic albedo image. "
        "Preserve the exact scene layout, visible objects, object boundaries, and material regions. "
        "Remove illumination effects only: cast shadows, attached shading, highlights, reflections, interreflections, exposure variation, ambient occlusion, and illumination color cast. "
        "Keep the intrinsic reflectance or base color of each visible surface. "
        "Preserve low-frequency and mid-frequency material texture that belongs to the surface itself, "
        "such as wood color variation, subtle fabric color variation, and printed material color, and material-region color boundaries, "
        "but do not preserve lighting-induced brightness gradients. "
        "Do not simplify the image into flat poster-like color blocks unless the original surface is truly uniform. "
        "Do not add new objects. Do not remove existing objects. Do not stylize. "
        "Output a clean, illumination-free, texture-preserving albedo map."
    )
    # analysis_block = "\n".join([
    #     "Analysis constraints:",
    #     f"Scene summary: {global_analysis.get('scene_summary', '') or 'N/A'}",
    #     f"Material notes: {material_notes or 'N/A'}",
    #     f"Lighting notes: {lighting_notes or 'N/A'}",
    #     f"View-specific hint: {per_view_hint or 'N/A'}",
    #     f"Additional constraints: {suffix or 'N/A'}",
    # ])

    analysis_block = "\n".join([
        "Analysis constraints:",
        f"Lighting notes: {lighting_notes or 'N/A'}",
        f"View-specific hint: {per_view_hint or 'N/A'}",
        f"Additional constraints: {suffix or 'N/A'}",
    ])

    hard_constraints = (
        "Hard constraints:\n"
        "This is a strictly structure-preserving appearance transformation task, not image redesign, scene completion, semantic reinterpretation, or content generation.\n"
        "Only remove illumination-dependent appearance from existing visible content.\n"
        "Preserve the scene content exactly.\n"
        "Do not create any new local structure patch anywhere, and do not delete any existing local structure patch, even if tiny.\n"
        "Do not change the count, order, spacing, or arrangement of small local entities on any surface.\n"
        "Do not convert weak or ambiguous evidence (blur, noise, reflection, low contrast) into clearer or more complete structure.\n"
        "Do not add, remove, replace, move, reorder, merge, split, or complete any visible entity, object part, material region, local structure, boundary, contour, or image region.\n"
        "Do not add any new object or element.\n"
        "Do not replace any existing object, region, or part with a different one.\n"
        "Do not create any new visible element, even if it would be plausible, aesthetically reasonable, or consistent with the scene.\n"
        "Do not delete or simplify any existing visible element, even if it is small, thin, partially occluded, low-contrast, blurry, reflective, saturated, dark, bright, or ambiguous.\n"
        "Do not reinterpret uncertain regions into clearer or more complete content.\n"
        "Do not infer or hallucinate hidden, missing, or implied structure.\n"
        "Do not modify spatial arrangement in any way: no change of position, alignment, scale, support relation, contact relation, overlap, depth ordering, or occlusion relation.\n"
        "Do not modify topology in any way: no new disconnected parts, no removed parts, no merged parts, and no split parts.\n"
        "Do not introduce new edges, new corners, new closed regions, new foreground/background elements, or new texture structures that are not already visibly present in the input.\n"
        "Do not convert weak visual evidence into stronger semantic content.\n"
        "Treat every uncertain region conservatively: preserve the original visible structure instead of inventing, cleaning up, rearranging, or completing it.\n"
        "The only allowed change is converting illumination-dependent appearance into illumination-free albedo while keeping all visible content and structure unchanged."
    )
    print( f"{base}\n\n"
        f"{analysis_block}\n\n"
        f"{hard_constraints}")
    return (
        f"{base}\n\n"
        f"{analysis_block}\n\n"
        f"{hard_constraints}"
    )


def run_gpt_image_generation(
    client: Any,
    model: str,
    prompt: str,
    source_image_path: Path,
    generation_mode: str,
    requested_size_arg: str,
    generate_requires_image: bool,
    quality: str,
    seed: int,
) -> Tuple[Any, str, Optional[Tuple[int, int]]]:
    mode = (generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")
    requested_size = normalize_gpt_image_size(requested_size_arg)
    common_kwargs: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "output_format": "png",
        "n": 1,
    }
    image_size = get_image_size(source_image_path)
    if mode == "generate":
        generate_kwargs = {**common_kwargs, "size": requested_size}
        # Prefer image-conditioned generation when supported.
        with source_image_path.open("rb") as image_file:
            try:
                return client.images.generate(image=image_file, seed=seed, **generate_kwargs), requested_size, image_size
            except TypeError:
                pass
        try:
            return client.images.generate(images=[{"image_url": file_to_data_uri(source_image_path)}], seed=seed, **generate_kwargs), requested_size, image_size
        except TypeError:
            if generate_requires_image:
                raise RuntimeError("Generate mode could not pass the image argument successfully; aborting to avoid a fallback to text-only generation.")
            try:
                return client.images.generate(**generate_kwargs), requested_size, image_size
            except TypeError:
                fallback_kwargs = dict(generate_kwargs)
                fallback_kwargs.pop("output_format", None)
                fallback_kwargs.pop("n", None)
                return client.images.generate(**fallback_kwargs), requested_size, image_size
    with source_image_path.open("rb") as image_file:
        try:
            return client.images.edit(image=image_file, size=requested_size, **common_kwargs), requested_size, image_size
        except TypeError:
            pass
    edit_kwargs = {
        **common_kwargs,
        "size": requested_size,
        "images": [{"image_url": file_to_data_uri(source_image_path)}],
    }
    try:
        return client.images.edit(**edit_kwargs), requested_size, image_size
    except TypeError:
        fallback_kwargs = dict(edit_kwargs)
        fallback_kwargs.pop("output_format", None)
        fallback_kwargs.pop("n", None)
        return client.images.edit(**fallback_kwargs), requested_size, image_size


def build_meta_paths(meta_dir: Path, num_parts: int, part_index: int) -> Tuple[Path, Path]:
    analysis_name = "multiview_analysis.json"
    manifest_name = "manifest.json"
    if num_parts > 1:
        suffix = f".part{part_index + 1}of{num_parts}"
        analysis_name = f"multiview_analysis{suffix}.json"
        manifest_name = f"manifest{suffix}.json"
    return meta_dir / analysis_name, meta_dir / manifest_name


def build_image_output_path(image_path: Path, input_dir: Path, base_dir: Path, suffix: str, preserve_relative_dirs: bool) -> Path:
    if preserve_relative_dirs:
        relative_path = image_path.relative_to(input_dir)
        return base_dir / relative_path.parent / f"{image_path.stem}{suffix}"
    return base_dir / f"{image_path.stem}{suffix}"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_pending_images(image_paths: List[Path], input_dir: Path, albedo_dir: Path, overwrite: bool, preserve_relative_dirs: bool) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending = []
    for image_path in image_paths:
        out_albedo = build_image_output_path(image_path, input_dir, albedo_dir, "_albedo.png", preserve_relative_dirs)
        if is_completed_output(out_albedo):
            continue
        pending.append(image_path)
    return pending


def process_images_with_external_analysis(
    args: argparse.Namespace,
    image_paths_for_generate: List[Path],
    pending_image_paths: List[Path],
    input_dir: Path,
    output_dir: Path,
    albedo_dir: Path,
    meta_dir: Path,
    image_client: Any,
    albedo_deployment: str,
    client_kind: str,
    endpoint: Optional[str],
    api_version: Optional[str],
    gt_root: Path,
    intro_text: str,
) -> None:
    analysis_sources = build_analysis_sources(args.analysis_dirs)
    if not analysis_sources:
        raise FileNotFoundError("No usable analysis directory was found. Provide at least one valid per_image_analysis directory via --analysis_dirs.")
    print(intro_text)
    print(
        "      "
        f"input_dir={input_dir.as_posix()} | "
        f"gt_root={gt_root.as_posix()} | "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"albedo_model={args.albedo_model} | "
        f"albedo_deployment={albedo_deployment} | "
        f"generation_mode={args.generation_mode} | "
        f"albedo_quality={args.albedo_quality} | "
        f"client_kind={client_kind} | "
        f"endpoint={endpoint or 'default'} | "
        f"api_version={api_version or 'default'} | "
        f"analysis_dirs={len(analysis_sources)}"
    )
    _, manifest_path = build_meta_paths(meta_dir, args.num_parts, args.part_index)
    skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)
    print("[2/3] Skip upstream model analysis and load the precomputed per-image analysis JSON files directly.")
    print(f"[3/3] Start exporting albedo maps ({len(pending_image_paths)} images)")
    if not pending_image_paths:
        print("  - The current shard is already complete; exiting.")
    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_albedo = build_image_output_path(image_path, input_dir, albedo_dir, "_albedo.png", args.preserve_relative_dirs)
        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(
                f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} "
                f"previously skipped, reason={skipped_entry.get('error_code', 'N/A')}"
            )
            continue
        if not args.overwrite and is_completed_output(out_albedo):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} already exists; skipping")
            continue
        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")
        try:
            single_analysis, analysis_path = load_external_analysis(image_path, input_dir, analysis_sources)
            print(f"      Loaded analysis: {analysis_path.as_posix()}")
        except Exception as exc:
            print(f"      Skip: failed to load the matched analysis result -> {exc}")
            record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "analysis_lookup", exc, args)
            continue
        try:
            source_image_path = resolve_gt_image_path(image_path, input_dir, gt_root)
        except Exception as exc:
            print(f"      Skip: failed to locate the GT source image -> {exc}")
            record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "gt_lookup", exc, args)
            continue
        per_view_hints = get_per_view_hints(single_analysis)
        hints = per_view_hints.get(image_path.name, {})
        albedo_prompt = make_albedo_prompt(single_analysis, hints.get("albedo_hint", ""))
        last_generation_exc: Optional[Exception] = None
        albedo_resp = None
        requested_size = None
        input_size = None
        for attempt in range(max(0, args.request_retries) + 1):
            try:
                albedo_resp, requested_size, input_size = run_gpt_image_generation(
                    image_client,
                    albedo_deployment,
                    albedo_prompt,
                    source_image_path,
                    args.generation_mode,
                    args.albedo_size,
                    args.generate_requires_image,
                    args.albedo_quality,
                    args.seed,
                )
                break
            except Exception as exc:
                last_generation_exc = exc
                if should_skip_image_error(exc):
                    print(f"      Skip: generation stage triggered content moderation -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                    break
                if should_retry_image_error(exc) and attempt < max(0, args.request_retries):
                    wait_seconds = min(args.retry_backoff * (2 ** attempt), args.retry_max_backoff)
                    print(
                        f"      Generation error detected; preparing retry "
                        f"({attempt + 1}/{max(0, args.request_retries)}) -> "
                        f"{exc.__class__.__name__}: {exc} | wait={wait_seconds:.1f}s"
                    )
                    time.sleep(max(0.0, wait_seconds))
                    continue
                if should_retry_image_error(exc):
                    print(f"      Skip: generation still failed after retries -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation_retry_exhausted", exc, args)
                    break
                raise
        if albedo_resp is None:
            if last_generation_exc is None:
                continue
            if should_skip_image_error(last_generation_exc) or should_retry_image_error(last_generation_exc):
                continue
            raise last_generation_exc
        ensure_parent_dir(out_albedo)
        save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)
        upsert_manifest_entry(manifest, {
            "image_name": image_path.name,
            "relative_image_path": image_path.relative_to(input_dir).as_posix(),
            "num_parts": args.num_parts,
            "part_index": args.part_index,
            "analysis_disabled": False,
            "analysis_model": "external_json",
            "analysis_deployment": "external_json",
            "albedo_model": args.albedo_model,
            "albedo_deployment": albedo_deployment,
            "client_kind": client_kind,
            "endpoint": endpoint,
            "api_version": api_version,
            "generation_mode": args.generation_mode,
            "requested_size": requested_size,
            "input_size": list(input_size) if input_size else None,
            "source_image_path": source_image_path.as_posix(),
            "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
            "analysis_output": analysis_path.as_posix(),
            "albedo_prompt": albedo_prompt,
            "albedo_hint": hints.get("albedo_hint", ""),
            "scene_summary": single_analysis.get("scene_summary", ""),
            "albedo_material_notes": single_analysis.get("albedo_material_notes", []),
            "albedo_lighting_notes": single_analysis.get("albedo_lighting_notes", []),
        })
        save_manifest(manifest_path, manifest)
        time.sleep(max(0.0, args.sleep))
    save_manifest(manifest_path, manifest)


def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    image_client, albedo_deployment, client_kind, endpoint, api_version = resolve_image_client_config(args)
    gt_root = Path(args.gt_root)
    requested_input_dir = Path(args.input_dir)
    input_dir = resolve_listing_input_dir(requested_input_dir, gt_root)
    output_dir = Path(args.output_dir)
    albedo_dir = output_dir / "albedo"
    meta_dir = output_dir / "meta"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate_all = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    image_paths_for_generate = shard_paths(image_paths_for_generate_all, args.num_parts, args.part_index)
    pending_image_paths = get_pending_images(
        image_paths_for_generate,
        input_dir,
        albedo_dir,
        args.overwrite,
        args.preserve_relative_dirs,
    )
    shard_label = f"part {args.part_index + 1}/{args.num_parts}"
    intro_text = (
        f"[1/3] Found {len(image_paths)} images, "
        f"current shard {shard_label} owns {len(image_paths_for_generate)} images, with {len(pending_image_paths)} pending, "
        "load precomputed per-image analysis and run GPT Image 1.5 generation or editing"
    )
    if requested_input_dir != input_dir:
        print(
            f"[info] Detected a migrated input_dir and remapped it automatically: "
            f"{requested_input_dir.as_posix()} -> {input_dir.as_posix()}"
        )
    process_images_with_external_analysis(
        args=args,
        image_paths_for_generate=image_paths_for_generate,
        pending_image_paths=pending_image_paths,
        input_dir=input_dir,
        output_dir=output_dir,
        albedo_dir=albedo_dir,
        meta_dir=meta_dir,
        image_client=image_client,
        albedo_deployment=albedo_deployment,
        client_kind=client_kind,
        endpoint=endpoint,
        api_version=api_version,
        gt_root=gt_root,
        intro_text=intro_text,
    )
    print("Done.")
    print(f"Albedo output directory: {albedo_dir.as_posix()}")
    print(f"Metadata output directory: {meta_dir.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
