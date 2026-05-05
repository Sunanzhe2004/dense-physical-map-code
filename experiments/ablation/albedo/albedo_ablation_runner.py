#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ANALYSIS_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_ALBEDO_MODEL = "doubao-seedream-5-0-260128"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FULL_PROMPT_TEXT = (
    "Convert the input image into a clean intrinsic albedo image. "
    "Preserve the exact scene layout, visible objects, object boundaries, and material regions. "
    "Remove illumination effects only: cast shadows, attached shading, highlights, reflections, "
    "interreflections, exposure variation, ambient occlusion, and illumination color cast. "
    "Keep the intrinsic reflectance or base color of each visible surface. "
    "Preserve low-frequency and mid-frequency material texture that belongs to the surface itself, "
    "such as wood color variation, subtle fabric color variation, printed material color, and "
    "material-region color boundaries, but do not preserve lighting-induced brightness gradients. "
    "Do not simplify the image into flat poster-like color blocks unless the original surface is truly uniform. "
    "Do not add new objects. Do not remove existing objects. Do not stylize. "
    "Output a clean, illumination-free, texture-preserving albedo map."
)

WEAKENED_PROMPT_TEXT = (
    "Convert the input indoor RGB image into an albedo image. "
    "Keep the scene geometry, objects, boundaries, and material colors. "
    "Reduce shadows, highlights, reflections, exposure changes, and other lighting effects. "
    "Preserve visible surface texture and do not add or remove objects."
)

MINIMAL_PROMPT_TEXT = "Generate an albedo map from the input image."


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    prompt_version: str
    description: str
    prompt_text: str
    analysis_mode: str = "none"


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_suffix", type=str, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input_dir.")
    parser.add_argument("--api_key", type=str, default=None, help="Provider API key. Falls back to ARK_API_KEY.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Provider base URL.")
    parser.add_argument(
        "--analysis_model",
        type=str,
        default=DEFAULT_ANALYSIS_MODEL,
        help="Analysis model used by analysis-conditioned variants.",
    )
    parser.add_argument("--albedo_model", type=str, default=DEFAULT_ALBEDO_MODEL, help="Image generation model.")
    parser.add_argument(
        "--max_views",
        type=int,
        default=8,
        help="Deprecated compatibility argument retained for older commands.",
    )
    parser.add_argument(
        "--analysis_max_side",
        type=int,
        default=1024,
        help="Resize the long side before analysis for analysis-conditioned variants.",
    )
    parser.add_argument(
        "--detail",
        type=str,
        default="high",
        choices=["low", "high", "auto"],
        help="Vision detail level for analysis-conditioned variants.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--albedo_size", type=str, default="adaptive")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs and metadata. By default completed outputs are skipped.",
    )
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)
    parser.add_argument(
        "--independent_images",
        action="store_true",
        help="Deprecated compatibility argument retained for older commands.",
    )
    parser.add_argument(
        "--preserve_relative_dirs",
        action="store_true",
        help="Preserve subdirectory layout relative to input_dir in output paths.",
    )
    parser.add_argument(
        "--analysis_scope",
        type=str,
        default="full",
        choices=["full", "part"],
        help="Deprecated compatibility argument retained for older commands.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = cli_api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Provide --api_key or set ARK_API_KEY.")
    return api_key


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
        recursive_msg = "recursively" if recursive else "non-recursively"
        raise FileNotFoundError(f"No supported image files{suffix_msg} found {recursive_msg}: {input_dir}")
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


def pil_resize_bytes(path: Path, max_side: int) -> Tuple[bytes, str]:
    if Image is None:
        return path.read_bytes(), guess_mime(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        long_side = max(width, height)
        if long_side > max_side:
            scale = max_side / float(long_side)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), "image/jpeg"


def file_to_data_uri(path: Path, max_side: Optional[int] = None) -> str:
    if max_side is None:
        data = path.read_bytes()
        mime = guess_mime(path)
    else:
        data, mime = pil_resize_bytes(path, max_side=max_side)
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json_file(path)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str) -> None:
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
    for pattern in (r"'code':\s*'([^']+)'", r'"code":\s*"([^"]+)"'):
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
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int = 120) -> None:
    url = getattr(image_item, "url", None)
    if url:
        save_url_to_file(url, save_path, timeout=timeout)
        return

    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        save_path.write_bytes(base64.b64decode(b64_json))
        return

    if isinstance(image_item, dict):
        if image_item.get("url"):
            save_url_to_file(image_item["url"], save_path, timeout=timeout)
            return
        if image_item.get("b64_json"):
            save_path.write_bytes(base64.b64decode(image_item["b64_json"]))
            return

    raise RuntimeError("Image response has neither url nor b64_json.")


def normalize_size(size: str) -> str:
    normalized = (size or "").strip().lower()
    if not normalized or normalized == "adaptive":
        return "2k"
    return normalized


def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_path: Path,
    size: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
) -> Any:
    image_data_uri = file_to_data_uri(image_path)
    normalized_size = normalize_size(size)
    model_lower = model.lower()

    if "seedream" in model_lower:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=[image_data_uri],
            size=normalized_size,
            watermark=watermark,
        )

    try:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_data_uri,
            seed=seed,
            guidance_scale=guidance_scale,
            size=normalized_size,
            watermark=watermark,
        )
    except TypeError:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_data_uri,
            size=normalized_size,
            watermark=watermark,
        )


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
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts)

    choices = getattr(resp, "choices", None)
    if choices:
        texts = []
        for choice in choices:
            message = getattr(choice, "message", None)
            if not message:
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str):
                texts.append(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        texts.append(part["text"])
                    else:
                        text = getattr(part, "text", None)
                        if text:
                            texts.append(text)
        if texts:
            return "\n".join(texts)

    return str(resp)


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return cleaned


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


def build_single_image_analysis_prompt(image_name: str) -> str:
    return f"""
You are given one indoor RGB image. Your task is not to generate an image. Instead, produce concise textual guidance that will help a later albedo-generation step.

Requirements:
1. Use only the visible evidence in this single image.
2. Focus on stable intrinsic material color cues rather than speculative scene details.
3. Distinguish true surface color from lighting effects such as shadows, highlights, reflections, exposure shifts, and illumination color casts.
4. Mention textures that belong to the material itself and should be preserved in albedo.
5. Do not invent details for hidden or ambiguous regions.
6. Output JSON only.

Input image name:
- {image_name}

Return exactly this JSON structure:
{{
  "scene_summary": "one-sentence scene summary",
  "albedo_material_notes": ["2-6 concise notes about intrinsic material color and texture"],
  "albedo_lighting_notes": ["2-6 concise notes about lighting artifacts that should be removed"],
  "albedo_prompt_suffix": "one short constraint sentence for albedo generation",
  "albedo_hint": "one short image-specific hint"
}}
""".strip()


def analyze_single_image(
    client: Any,
    image_path: Path,
    model: str,
    detail: str = "high",
    analysis_max_side: int = 1024,
) -> Dict[str, Any]:
    prompt = build_single_image_analysis_prompt(image_path.name)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_url: Dict[str, Any] = {"url": file_to_data_uri(image_path, max_side=analysis_max_side)}
    if detail in {"low", "high"}:
        image_url["detail"] = detail
    content.append({"type": "image_url", "image_url": image_url})

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
    )
    data = robust_json_loads(extract_output_text(response))
    data.setdefault("scene_summary", "")
    data.setdefault("albedo_material_notes", [])
    data.setdefault("albedo_lighting_notes", [])
    data.setdefault("albedo_prompt_suffix", "")
    data.setdefault("albedo_hint", "")
    return data


def make_analysis_conditioned_prompt(base_prompt: str, single_analysis: Dict[str, Any]) -> str:
    material_notes = "; ".join(single_analysis.get("albedo_material_notes", []))
    lighting_notes = "; ".join(single_analysis.get("albedo_lighting_notes", []))
    prompt_suffix = single_analysis.get("albedo_prompt_suffix", "")
    albedo_hint = single_analysis.get("albedo_hint", "")
    return (
        f"{base_prompt}\n\n"
        f"Scene summary: {single_analysis.get('scene_summary', '') or 'N/A'}\n"
        f"Albedo material notes: {material_notes or 'N/A'}\n"
        f"Albedo lighting notes: {lighting_notes or 'N/A'}\n"
        f"Image-specific hint: {albedo_hint or 'N/A'}\n"
        f"Additional constraints: {prompt_suffix or 'N/A'}"
    )


def build_meta_paths(meta_dir: Path, num_parts: int, part_index: int, analysis_mode: str) -> Tuple[Path, Path]:
    analysis_dir = meta_dir / "per_image_analysis"
    manifest_name = "manifest.json"
    if num_parts > 1:
        manifest_name = f"manifest.part{part_index + 1}of{num_parts}.json"
    if analysis_mode == "single_image":
        return analysis_dir, meta_dir / manifest_name
    return meta_dir, meta_dir / manifest_name


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
    albedo_dir: Path,
    overwrite: bool,
    preserve_relative_dirs: bool,
) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending = []
    for image_path in image_paths:
        out_albedo = build_image_output_path(
            image_path,
            input_dir,
            albedo_dir,
            "_albedo.png",
            preserve_relative_dirs,
        )
        if is_completed_output(out_albedo):
            continue
        pending.append(image_path)
    return pending


def create_analysis_client(base_url: str, api_key: str) -> Any:
    if OpenAI is None:
        raise ImportError("Failed to import openai. Install with: pip install openai") from _OPENAI_IMPORT_ERROR
    return OpenAI(base_url=base_url, api_key=api_key)


def run_variant(config: VariantConfig) -> None:
    args = parse_args(config)
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key(args.api_key)
    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Install with: pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

    input_dir = Path(args.input_dir)
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
    analysis_client = create_analysis_client(args.base_url, api_key) if config.analysis_mode == "single_image" else None
    ark_client = Ark(base_url=args.base_url, api_key=api_key)

    analysis_dir, manifest_path = build_meta_paths(meta_dir, args.num_parts, args.part_index, config.analysis_mode)
    if config.analysis_mode == "single_image":
        analysis_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)

    print(
        f"[{config.variant_id}] {config.variant_name}: found {len(image_paths)} images, "
        f"{shard_label} owns {len(image_paths_for_generate)}, pending {len(pending_image_paths)}"
    )
    print(
        "      "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"analysis_mode={config.analysis_mode} | "
        f"analysis_model={args.analysis_model if config.analysis_mode == 'single_image' else 'N/A'} | "
        f"albedo_model={args.albedo_model} | "
        f"base_url={args.base_url}"
    )

    if not pending_image_paths:
        print("  - All outputs for this shard already exist.")

    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_albedo = build_image_output_path(
            image_path,
            input_dir,
            albedo_dir,
            "_albedo.png",
            args.preserve_relative_dirs,
        )
        analysis_output_path = build_image_output_path(
            image_path,
            input_dir,
            analysis_dir,
            ".json",
            args.preserve_relative_dirs,
        )
        skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
        if skipped_entry is not None:
            print(
                f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} "
                f"previously skipped, reason={skipped_entry.get('error_code', 'N/A')}"
            )
            continue
        if not args.overwrite and is_completed_output(out_albedo):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} exists, skip")
            continue

        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")

        single_analysis: Optional[Dict[str, Any]] = None
        prompt_text = config.prompt_text
        if config.analysis_mode == "single_image":
            if not args.overwrite and analysis_output_path.exists():
                single_analysis = load_json_file(analysis_output_path)
                print("      reusing existing per-image analysis")
            else:
                try:
                    single_analysis = analyze_single_image(
                        analysis_client,
                        image_path,
                        args.analysis_model,
                        args.detail,
                        args.analysis_max_side,
                    )
                except Exception as exc:
                    if should_skip_image_error(exc):
                        print(f"      skip: analysis triggered content filter -> {extract_error_code(exc)}")
                        record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "analysis", exc, args)
                        continue
                    raise
                ensure_parent_dir(analysis_output_path)
                analysis_output_path.write_text(
                    json.dumps(single_analysis, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            prompt_text = make_analysis_conditioned_prompt(config.prompt_text, single_analysis)

        try:
            albedo_resp = run_image_generation(
                ark_client,
                args.albedo_model,
                prompt_text,
                image_path,
                args.albedo_size,
                args.watermark,
                args.seed,
                args.guidance_scale,
            )
        except Exception as exc:
            if should_skip_image_error(exc):
                print(f"      skip: generation triggered content filter -> {extract_error_code(exc)}")
                record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                continue
            raise

        ensure_parent_dir(out_albedo)
        save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)
        relative_image_path = image_path.relative_to(input_dir).as_posix()
        manifest_entry: Dict[str, Any] = {
            "manifest_key": relative_image_path,
            "image_name": image_path.name,
            "relative_image_path": relative_image_path,
            "num_parts": args.num_parts,
            "part_index": args.part_index,
            "variant_id": config.variant_id,
            "variant_name": config.variant_name,
            "prompt_version": config.prompt_version,
            "analysis_mode": config.analysis_mode,
            "analysis_model": args.analysis_model if config.analysis_mode == "single_image" else None,
            "albedo_model": args.albedo_model,
            "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
            "prompt_text": prompt_text,
        }
        if single_analysis is not None:
            manifest_entry["analysis_output"] = analysis_output_path.relative_to(output_dir).as_posix()
            manifest_entry["albedo_hint"] = single_analysis.get("albedo_hint", "")
            manifest_entry["scene_summary"] = single_analysis.get("scene_summary", "")
            manifest_entry["albedo_material_notes"] = single_analysis.get("albedo_material_notes", [])
            manifest_entry["albedo_lighting_notes"] = single_analysis.get("albedo_lighting_notes", [])
        upsert_manifest_entry(manifest, manifest_entry, key_field="manifest_key")
        time.sleep(max(0.0, args.sleep))

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    print(f"albedo output dir: {albedo_dir.as_posix()}")
    print(f"meta output dir: {meta_dir.as_posix()}")
