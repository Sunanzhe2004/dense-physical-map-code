#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate albedo maps for multi-view indoor scenes.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_suffix", type=str, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan all subdirectories under input_dir.")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--analysis_model", type=str, default=DEFAULT_ANALYSIS_MODEL)
    parser.add_argument("--albedo_model", type=str, default=DEFAULT_ALBEDO_MODEL)
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--analysis_max_side", type=int, default=1024)
    parser.add_argument("--detail", type=str, default="high", choices=["low", "high", "auto"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--albedo_size", type=str, default="adaptive")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs and analysis files instead of skipping them.")
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)
    parser.add_argument("--independent_images", action="store_true", help="Treat each input image as an independent sample instead of sharing multi-view analysis.")
    parser.add_argument("--preserve_relative_dirs", action="store_true", help="Preserve the relative directory layout from input_dir in the output tree.")
    parser.add_argument("--analysis_scope", type=str, default="full", choices=["full", "part"], help="Use all views for analysis or analyze each shard independently.")
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

def extract_output_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return output_text
    if hasattr(resp, "output"):
        texts = []
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
        texts = []
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

def build_analysis_prompt(image_names: List[str]) -> str:
    names_text = "\n".join(f"- {name}" for name in image_names)
    return f"""
You will be shown multiple RGB views of the same indoor scene. Your task is not to generate an image. Instead, summarize the most reliable cross-view material and lighting evidence needed for downstream albedo generation.

Requirements:
1. Fuse information across all views and keep the result cross-view consistent.
2. Keep only the most stable and confident observations.
3. Focus on intrinsic material color versus illumination-driven brightness changes.
4. Highlight which bright or dark regions come from shadows, highlights, exposure, or reflections rather than true albedo.
5. Highlight which surfaces contain intrinsic texture that should be preserved, such as wood color variation, fabric prints, or decorative wall color.
6. Do not invent hidden or unseen regions.
7. Output JSON only, with no extra explanation.

Input view filenames:
{names_text}

Return exactly this JSON structure:
{{
  "scene_summary": "one-sentence scene summary",
  "albedo_material_notes": ["2 to 6 key material or intrinsic-color notes"],
  "albedo_lighting_notes": ["2 to 6 key lighting, shadow, or highlight notes"],
  "albedo_prompt_suffix": "one additional constraint emphasizing intrinsic texture preservation and illumination removal",
  "per_view": [
    {{
      "image_name": "must exactly match an input filename",
      "albedo_hint": "one short albedo-specific note for this view"
    }}
  ]
}}
""".strip()

def analyze_multiview(client: OpenAI, image_paths: List[Path], model: str, detail: str = "high", analysis_max_side: int = 1024) -> Dict[str, Any]:
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
        temperature=0.1,
    )
    text = extract_output_text(resp)
    data = robust_json_loads(text)
    data.setdefault("scene_summary", "")
    data.setdefault("albedo_material_notes", [])
    data.setdefault("albedo_lighting_notes", [])
    data.setdefault("albedo_prompt_suffix", "")
    data.setdefault("per_view", [])
    return data

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
    return (
        f"{base}\n\n"
        f"Scene summary: {global_analysis.get('scene_summary', '') or 'N/A'}\n"
        f"Albedo material notes: {material_notes or 'N/A'}\n"
        f"Albedo lighting notes: {lighting_notes or 'N/A'}\n"
        f"View-specific hint: {per_view_hint or 'N/A'}\n"
        f"Additional constraints: {suffix or 'N/A'}"
    )

def run_image_generation(ark_client: Any, model: str, prompt: str, image_path: Path, size: str, watermark: bool, seed: int, guidance_scale: float) -> Any:
    image_data_uri = file_to_data_uri(image_path, max_side=None)
    model_lower = model.lower()
    if size:
        size = size.strip().lower()
    if not size or size == "adaptive":
        size = "2k"
    if "seedream" in model_lower:
        return ark_client.images.generate(model=model, prompt=prompt, image=[image_data_uri], size=size, watermark=watermark)
    try:
        return ark_client.images.generate(model=model, prompt=prompt, image=image_data_uri, seed=seed, guidance_scale=guidance_scale, size=size, watermark=watermark)
    except TypeError:
        return ark_client.images.generate(model=model, prompt=prompt, image=image_data_uri, size=size, watermark=watermark)

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

def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key(args.api_key)
    if Ark is None:
        raise ImportError('Failed to import volcenginesdkarkruntime. Please install: pip install "volcengine-python-sdk[ark]"') from _ARK_IMPORT_ERROR
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
    oa_client = OpenAI(base_url=args.base_url, api_key=api_key)
    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    if args.independent_images:
        print(
            f"[1/3] Independent-image mode: found {len(image_paths)} images, "
            f"current shard {shard_label} owns {len(image_paths_for_generate)} images, with {len(pending_image_paths)} pending"
        )
        print(
            "      "
            f"filename_suffix={args.filename_suffix or 'N/A'} | "
            f"recursive={args.recursive} | "
            f"preserve_relative_dirs={args.preserve_relative_dirs} | "
            f"analysis_model={args.analysis_model} | "
            f"albedo_model={args.albedo_model} | "
            f"base_url={args.base_url}"
        )
        analysis_dir = meta_dir / "per_image_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _, manifest_path = build_meta_paths(meta_dir, args.num_parts, args.part_index)
        skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
        manifest = load_manifest(manifest_path)
        skipped_images = load_manifest(skipped_path)
        print(f"[2/3] Skip global multi-view analysis and process each image independently.")
        print(f"[3/3] Start exporting albedo maps ({len(pending_image_paths)} images)")
        if not pending_image_paths:
            print("  - The current shard is already complete; exiting.")
        for idx, image_path in enumerate(image_paths_for_generate, start=1):
            out_albedo = build_image_output_path(image_path, input_dir, albedo_dir, "_albedo.png", args.preserve_relative_dirs)
            single_analysis_path = build_image_output_path(image_path, input_dir, analysis_dir, ".json", args.preserve_relative_dirs)
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
            if not args.overwrite and single_analysis_path.exists():
                single_analysis = load_json_file(single_analysis_path)
                print("      Reusing existing per-image analysis")
            else:
                try:
                    single_analysis = analyze_multiview(oa_client, [image_path], args.analysis_model, args.detail, args.analysis_max_side)
                except Exception as exc:
                    if should_skip_image_error(exc):
                        print(f"      Skip: analysis stage triggered content moderation -> {extract_error_code(exc)}")
                        record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "analysis", exc, args)
                        continue
                    raise
                ensure_parent_dir(single_analysis_path)
                single_analysis_path.write_text(json.dumps(single_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
            per_view_hints = get_per_view_hints(single_analysis)
            hints = per_view_hints.get(image_path.name, {})
            albedo_prompt = make_albedo_prompt(single_analysis, hints.get("albedo_hint", ""))
            try:
                albedo_resp = run_image_generation(ark_client, args.albedo_model, albedo_prompt, image_path, args.albedo_size, args.watermark, args.seed, args.guidance_scale)
            except Exception as exc:
                if should_skip_image_error(exc):
                    print(f"      Skip: generation stage triggered content moderation -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                    continue
                raise
            ensure_parent_dir(out_albedo)
            save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)
            upsert_manifest_entry(manifest, {
                "image_name": image_path.name,
                "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                "num_parts": args.num_parts,
                "part_index": args.part_index,
                "analysis_model": args.analysis_model,
                "albedo_model": args.albedo_model,
                "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
                "analysis_output": single_analysis_path.relative_to(output_dir).as_posix(),
                "albedo_prompt": albedo_prompt,
                "albedo_hint": hints.get("albedo_hint", ""),
                "scene_summary": single_analysis.get("scene_summary", ""),
                "albedo_material_notes": single_analysis.get("albedo_material_notes", []),
                "albedo_lighting_notes": single_analysis.get("albedo_lighting_notes", []),
            })
            time.sleep(max(0.0, args.sleep))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if args.analysis_scope == "full":
            analysis_candidates = image_paths
        else:
            analysis_candidates = shard_paths(image_paths, args.num_parts, args.part_index)
        image_paths_for_analysis = analysis_candidates[: max(1, args.max_views)]
        print(
            f"[1/3] Found {len(image_paths)} images, "
            f"current shard {shard_label} owns {len(image_paths_for_generate)} images, with {len(pending_image_paths)} pending, "
            f"using {len(image_paths_for_analysis)}  views for shared multi-view analysis"
        )
        print(
            "      "
            f"filename_suffix={args.filename_suffix or 'N/A'} | "
            f"recursive={args.recursive} | "
            f"preserve_relative_dirs={args.preserve_relative_dirs} | "
            f"analysis_model={args.analysis_model} | "
            f"albedo_model={args.albedo_model} | "
            f"base_url={args.base_url}"
        )
        analysis_path, manifest_path = build_meta_paths(meta_dir, args.num_parts, args.part_index)
        skipped_path = build_skip_path(meta_dir, args.num_parts, args.part_index)
        manifest = load_manifest(manifest_path)
        skipped_images = load_manifest(skipped_path)
        if not pending_image_paths:
            print("[2/3] The current shard is already complete; skipping analysis and generation.")
            print("Done.")
            print(f"Albedo output directory: {albedo_dir.as_posix()}")
            print(f"Metadata output directory: {meta_dir.as_posix()}")
            return
        if not args.overwrite and analysis_path.exists():
            global_analysis = load_json_file(analysis_path)
            print(f"[2/3] Reusing existing multi-view analysis: {analysis_path.as_posix()}")
        else:
            global_analysis = analyze_multiview(oa_client, image_paths_for_analysis, args.analysis_model, args.detail, args.analysis_max_side)
            analysis_path.write_text(json.dumps(global_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[2/3] Multi-view analysis completed and saved: {analysis_path.as_posix()}")
        per_view_hints = get_per_view_hints(global_analysis)
        print(f"[3/3] Start exporting albedo maps ({len(pending_image_paths)} images)")
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
            hints = per_view_hints.get(image_path.name, {})
            albedo_prompt = make_albedo_prompt(global_analysis, hints.get("albedo_hint", ""))
            try:
                albedo_resp = run_image_generation(ark_client, args.albedo_model, albedo_prompt, image_path, args.albedo_size, args.watermark, args.seed, args.guidance_scale)
            except Exception as exc:
                if should_skip_image_error(exc):
                    print(f"      Skip: generation stage triggered content moderation -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc, args)
                    continue
                raise
            ensure_parent_dir(out_albedo)
            save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)
            upsert_manifest_entry(manifest, {
                "image_name": image_path.name,
                "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                "num_parts": args.num_parts,
                "part_index": args.part_index,
                "analysis_model": args.analysis_model,
                "albedo_model": args.albedo_model,
                "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
                "albedo_prompt": albedo_prompt,
                "albedo_hint": hints.get("albedo_hint", ""),
                "scene_summary": global_analysis.get("scene_summary", ""),
                "albedo_material_notes": global_analysis.get("albedo_material_notes", []),
                "albedo_lighting_notes": global_analysis.get("albedo_lighting_notes", []),
            })
            time.sleep(max(0.0, args.sleep))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    print(f"Albedo output directory: {albedo_dir.as_posix()}")
    print(f"Metadata output directory: {meta_dir.as_posix()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
