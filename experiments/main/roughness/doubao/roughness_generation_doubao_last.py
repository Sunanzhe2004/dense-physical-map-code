#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main pipeline: direct roughness generation from RGB plus an externally provided
# segmentation prior. In our experiments, the prior masks are SAM3 outputs, but
# this script only consumes the prior and does not generate segmentation itself.

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_IMAGE_MODEL = "doubao-seedream-5-0-260128"
DEFAULT_TIMEOUT = 1800
PROMPT_VERSION = "capability_rgbonly_weakprior_v1"
SEG_CANDIDATE_SUFFIXES = ["_seg", "_mask", "_sam", "_semantic", "_label", ""]
INPUT_MODE_CHOICES = ("rgb_plus_seg", "rgb_only", "rgb_plus_example")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate roughness maps with SeedDream from RGB-only, RGB+example, or RGB+seg inputs."
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
        default="rgb_only",
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
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Ark base URL")
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument("--size", type=str, default="adaptive", help="Output size; adaptive -> 2k")
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


def ensure_api_key() -> str:
    api_key = os.environ.get("ARK_API_KEY")
    if api_key:
        api_key = api_key.strip()
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                "ARK_API_KEY contains non-ASCII characters and cannot be written to the HTTP header. "
                "Please replace any placeholder text with the real key."
            ) from e
        return api_key
    raise RuntimeError("Missing authentication: please provide ARK_API_KEY.")


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
        seg_name = "ObjectSegmentation_" + rgb_path.name[len("Image_"):]
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
            "You are a senior PBR material analysis expert. "
            "You are given exactly one input image: the target RGB image. "
            "Generate only the target roughness map for that RGB image. "
            "The output must be a single-channel grayscale roughness map spatially aligned with the target RGB content. "
            "Preserve the exact scene layout and object presence from the target RGB image. "
            "Do not add, remove, replace, duplicate, move, or reshape objects, object parts, or visible scene structures. "
            "Do not hallucinate missing content and do not erase existing content. "
            "Black means very smooth or polished. White means very rough and diffuse. "
            "Roughness ranges: 0.0-0.1 means mirror-like or highly polished, 0.1-0.3 means smooth or glossy, 0.3-0.6 means semi-gloss to moderately rough, 0.6-0.85 means rough diffuse material, and 0.85-1.0 means very rough, heavily scattering surfaces. "
            "Output one image only: no color, no text, no labels, no collage, and no overlay. "
            "Do not copy RGB brightness, direct illumination, cast shadows, self-shadowing, ambient occlusion, reflections, or bright highlights into roughness. "
            "Do not convert low-frequency lighting gradients into low-frequency roughness gradients. "
            "If the same material appears under different lighting, keep the roughness similar across those pixels even if the RGB brightness changes strongly. "
            "Infer roughness from material appearance, highlight sharpness, reflection behavior, coating cues, and visible surface micro-structure cues, not from brightness alone. "
            "Albedo patterns, color changes, printed textures, and semantic identity do not automatically imply strong roughness variation. "
            "Estimate roughness from the visible evidence in the image without forcing unnecessary smoothness, segmentation-like regions, or dataset-specific texture statistics. "
            "Allow local roughness variation when the image supports it, but do not invent unsupported fine detail. "
            "If the evidence is ambiguous, make the most plausible roughness estimate from the image rather than collapsing large areas to a default constant tone. "
            "Do not stylize the result. "
            "Output exactly one grayscale roughness map only."
        )

    if input_mode == "rgb_plus_example":
        return (
            "You are a senior PBR material analysis expert. "
            "You are given exactly three input images in order: "
            "(1) reference RGB image, "
            "(2) reference roughness map, "
            "(3) target RGB image. "
            "Generate only the target roughness map for image (3). "
            "Use image (2) only as dataset-style reference for output format, tone distribution, and roughness-map appearance. "
            "Do not copy the reference scene layout, object arrangement, or semantic material identity onto the target scene. "
            "The output must be a single-channel grayscale roughness map spatially aligned with the target RGB content. "
            "Black means very smooth or polished. White means very rough and diffuse. "
            "Roughness ranges: 0.0-0.1 means mirror-like or highly polished, 0.1-0.3 means smooth or glossy, 0.3-0.6 means semi-gloss to moderately rough, 0.6-0.85 means rough diffuse material, and 0.85-1.0 means very rough, heavily scattering surfaces. "
            "Output one image only: no color, no text, no labels, no collage, and no overlay. "
            "Do not copy RGB brightness, direct illumination, cast shadows, self-shadowing, ambient occlusion, reflections, or bright highlights into roughness. "
            "Do not convert low-frequency lighting gradients into low-frequency roughness gradients. "
            "If the same material appears under different lighting, keep the roughness similar across those pixels even if the RGB brightness changes strongly. "
            "Infer roughness from material appearance, highlight sharpness, reflection behavior, coating cues, and visible surface micro-structure cues, not from brightness alone. "
            "Use the reference pair as output-style guidance only, not as a material template for the target. "
            "Prefer a piecewise-smooth material map over a photometric grayscale rendering. "
            "Large homogeneous surfaces should usually remain spatially stable when the RGB evidence supports that. "
            "Albedo patterns, color changes, printed textures, and semantic identity do not automatically imply strong roughness variation. "
            "Local roughness variation should appear only when the visible material micro-structure clearly supports it; shading bands and lighting gradients must not dominate the map. "
            "When uncertain, prefer conservative and spatially stable roughness estimates rather than exaggerated local contrast. "
            "Do not stylize the result. Do not invent high-frequency texture where the RGB image does not support it. "
            "Output exactly one grayscale roughness map only."
        )

    return (
        "You are a senior PBR material analysis expert. "
        "You are given exactly two input images in order: "
        "(1) target RGB image, "
        "(2) target segmentation prior map. "
        "Generate only the target roughness map for the RGB image. "
        "Use the segmentation map as a soft spatial prior for object and material boundaries, not as a guaranteed material segmentation. "
        "A single segmented object may contain multiple materials, and disconnected or visually heterogeneous regions should not be forced to share one roughness value. "
        "Preserve material changes supported by the RGB image even when they fall inside the same segmentation region. "
        "Use segmentation mainly to stabilize boundaries and large spatial layout, not to force one scalar roughness per region. "
        "The output must be a single-channel grayscale roughness map spatially aligned with the target RGB content. "
        "Black means very smooth or polished. White means very rough and diffuse. "
        "Roughness ranges: 0.0-0.1 means mirror-like or highly polished, 0.1-0.3 means smooth or glossy, 0.3-0.6 means semi-gloss to moderately rough, 0.6-0.85 means rough diffuse material, and 0.85-1.0 means very rough, heavily scattering surfaces. "
        "Output one image only: no color, no text, no labels, no collage, and no overlay. "
        "Do not copy RGB brightness, direct illumination, cast shadows, self-shadowing, ambient occlusion, reflections, or bright highlights into roughness. "
        "Do not convert low-frequency lighting gradients into low-frequency roughness gradients. "
        "If the same material appears under different lighting, keep the roughness similar across those pixels even if the RGB brightness changes strongly. "
        "Infer roughness from material appearance, highlight sharpness, reflection behavior, coating cues, and visible surface micro-structure cues, not from brightness alone. "
        "Keep object and material boundaries spatially coherent when supported by visible material evidence, but do not over-trust coarse segmentation when it conflicts with clear RGB material evidence. "
        "Prefer a piecewise-smooth material map over a photometric grayscale rendering. "
        "Large homogeneous surfaces should usually remain spatially stable, but allow RGB-supported finish changes within coarse object masks. "
        "Albedo patterns, color changes, printed textures, and semantic identity do not automatically imply strong roughness variation. "
        "Local roughness variation should appear only when the visible material micro-structure clearly supports it; shading bands and lighting gradients must not dominate the map. "
        "When uncertain, prefer conservative and spatially stable roughness estimates rather than exaggerated local contrast, without forcing a single constant value inside every segmentation region. "
        "Do not stylize the result. Do not invent high-frequency texture where the RGB image does not support it. "
        "Output exactly one grayscale roughness map only."
    )


def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
) -> Any:
    image_uris = [file_to_data_uri(p) for p in image_paths]
    model_lower = model.lower()

    if size:
        size = size.strip().lower()
    if not size or size == "adaptive":
        size = "2k"

    if "seedream" in model_lower:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris,
            size=size,
            watermark=watermark,
        )

    return ark_client.images.generate(
        model=model,
        prompt=prompt,
        image=image_uris,
        size=size,
        watermark=watermark,
    )


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
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

    raise RuntimeError("Image response contains neither url nor b64_json.")


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
    watermark: bool,
) -> Dict[str, Any]:
    if input_mode == "rgb_only":
        route = "seedream_rgb_only"
    elif input_mode == "rgb_plus_example":
        route = "seedream_rgb_plus_example"
    else:
        route = "seedream_rgb_sam3_soft_direct"

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
        "output_resolution_policy": "model_native_output",
        "watermark": watermark,
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


def generate_roughness_map_with_seedream(
    *,
    ark_client: Any,
    model: str,
    input_mode: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    save_path: Path,
    example_rgb: Optional[Path],
    example_roughness: Optional[Path],
    size: str,
    watermark: bool,
    timeout: int,
) -> str:
    if input_mode == "rgb_only":
        prompt = build_roughness_prompt(input_mode)
        image_paths = [rgb_path]
        mode = "seedream_rgb_only"
    elif input_mode == "rgb_plus_example":
        if not example_rgb or not example_roughness:
            raise ValueError("rgb_plus_example requires both --example_rgb and --example_roughness")
        prompt = build_roughness_prompt(input_mode)
        image_paths = [example_rgb, example_roughness, rgb_path]
        mode = "seedream_rgb_plus_example"
    else:
        if seg_path is None:
            raise ValueError("rgb_plus_seg requires a matching segmentation image")
        prompt = build_roughness_prompt(input_mode)
        image_paths = [rgb_path, seg_path]
        mode = "seedream_rgb_sam3_soft_direct"

    response = run_image_generation(
        ark_client=ark_client,
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        size=size,
        watermark=watermark,
    )
    save_image_response(response.data[0], save_path, timeout=timeout)
    enforce_grayscale_png(save_path)
    return mode


def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key()

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

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
        size=args.size,
        watermark=args.watermark,
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
        f"input_mode={args.input_mode}"
    )
    print(
        "      "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"overwrite={args.overwrite} | "
        f"base_url={args.base_url}"
    )

    ark_client = Ark(base_url=args.base_url, api_key=api_key)

    setup = dict(run_signature)
    setup.update(
        {
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
            item["roughness_mode"] = generate_roughness_map_with_seedream(
                ark_client=ark_client,
                model=args.image_model,
                input_mode=args.input_mode,
                rgb_path=rgb_path,
                seg_path=seg_path,
                save_path=roughness_path,
                example_rgb=example_rgb,
                example_roughness=example_roughness,
                size=args.size,
                watermark=args.watermark,
                timeout=args.timeout,
            )
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
