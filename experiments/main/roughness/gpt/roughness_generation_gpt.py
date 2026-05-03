#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Main pipeline: direct roughness generation from RGB plus an externally provided
# segmentation prior. In our experiments, the prior masks are SAM3 outputs, but
# this script only consumes the prior and does not generate segmentation itself.

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
PROMPT_VERSION = "capability_rgbonly_textureaware_v3"
SEG_CANDIDATE_SUFFIXES = ["_seg", "_mask", "_sam", "_semantic", "_label", ""]
INPUT_MODE_CHOICES = ("rgb_plus_seg", "rgb_only", "rgb_plus_example")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_GPT_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate roughness maps with GPT Image 1.5 from RGB-only, RGB+example, or RGB+seg inputs."
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
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI or Azure OpenAI API key")
    parser.add_argument(
        "--base_url",
        "--azure_endpoint",
        dest="base_url",
        type=str,
        default=os.environ.get("AZURE_ROUGHNESS_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or DEFAULT_BASE_URL,
        help="Azure OpenAI endpoint or OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=os.environ.get("AZURE_ROUGHNESS_OPENAI_API_VERSION")
        or os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or DEFAULT_API_VERSION,
        help="Azure OpenAI API version",
    )
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument("--size", type=str, default=DEFAULT_IMAGE_SIZE, help="GPT Image output size")
    parser.add_argument("--image_quality", type=str, default=DEFAULT_IMAGE_QUALITY, choices=["low", "medium", "high", "auto"])
    parser.add_argument("--watermark", action="store_true", help="Compatibility flag; ignored for GPT Image 1.5")
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
    parser.add_argument(
        "--save_debug_intermediates",
        action="store_true",
        help="Save full-canvas and cropped intermediate images for debugging padding/cropping drift.",
    )
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = (
        cli_api_key
        or os.environ.get("AZURE_ROUGHNESS_OPENAI_API_KEY")
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
        "AZURE_ROUGHNESS_OPENAI_API_KEY / AZURE_GPT_IMAGE_15_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY."
    )


def resolve_image_client_config(args: argparse.Namespace) -> Tuple[Any, str, str, Optional[str], Optional[str]]:
    api_key = ensure_api_key(args.api_key)
    endpoint = (
        args.base_url
        or os.environ.get("AZURE_ROUGHNESS_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    api_version = (
        args.api_version
        or os.environ.get("AZURE_ROUGHNESS_OPENAI_API_VERSION")
        or os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
    )
    deployment = args.image_model
    if endpoint:
        if AzureOpenAI is None:
            raise ImportError("Failed to import openai. Please install: pip install openai") from _OPENAI_IMPORT_ERROR
        client = AzureOpenAI(
            api_version=api_version or DEFAULT_API_VERSION,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        return client, deployment, "azure", endpoint, api_version or DEFAULT_API_VERSION
    if OpenAI is None:
        raise ImportError("Failed to import openai. Please install: pip install openai") from _OPENAI_IMPORT_ERROR
    client = OpenAI(api_key=api_key)
    return client, deployment, "openai", endpoint, api_version


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


def validate_parts(num_parts: int, part_index: int) -> None:
    if num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if part_index < 0 or part_index >= num_parts:
        raise ValueError("--part_index must satisfy 0 <= part_index < --num_parts")


def shard_paths(paths: List[Path], num_parts: int, part_index: int) -> List[Path]:
    if num_parts == 1:
        return list(paths)
    return [path for idx, path in enumerate(paths) if idx % num_parts == part_index]


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
        raise ValueError(f"bbox length must be 4, got: {bbox}")

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
            "Do not copy RGB brightness, direct illumination, cast shadows, self-shadowing, ambient occlusion, reflections, bright highlights, bloom, exposure falloff, haze, or vignetting into roughness. "
            "Do not convert low-frequency lighting gradients into low-frequency roughness gradients. "
            "Do not use darkness, shadow boundaries, bright highlights, or reflected scenery as direct grayscale templates for roughness. "
            "If the same material appears under different lighting, keep the roughness similar across those pixels even if RGB brightness changes strongly. "
            "Infer roughness from material appearance, highlight sharpness, reflection behavior, coating cues, and visible surface finish cues, not from brightness alone. "
            "Albedo patterns, color changes, printed textures, and semantic identity do not automatically imply strong roughness variation. "
            "Prefer a clean material-property map rather than a noisy grayscale rendering. "
            "Do not introduce grain, speckle, stippling, dithering, Monte-Carlo-like noise, film grain, sensor noise, or random pixel-level texture anywhere in the output. "
            "Within one visible material region, keep roughness spatially coherent, but preserve consistent local variation when the surface finish itself visibly shows repeated texture, weave, grain, pores, or brushed structure. "
            "Different visible materials should usually have distinguishable roughness values when the image supports that difference. "
            "Large homogeneous surfaces such as painted walls, ceilings, cabinet panels, large stone slabs, and broad floor areas should usually remain spatially stable, but do not force perfectly flat fills when visible finish variation is actually present. "
            "The map must not look washed out, foggy, milky, veiled, or globally low-contrast. "
            "Use a plausible material-dependent tonal spread: smoother materials should be noticeably darker than rough diffuse materials when evidence supports it. "
            "Do not compress most of the scene into a narrow value range. "
            "Do not treat dark color, black paint, shadow, backlighting, silhouettes, or low exposure as evidence of low roughness. "
            "Black-painted, dark-colored, or shadowed objects are not automatically smoother. "
            "Do not force mirrors, glass, screens, windows, or other dark-looking or reflective regions to pure black by default just because they appear dark or reflective in RGB. "
            "For mirrors, windows, glass, and screens, estimate the roughness of the surface itself, not the reflected scene, not the seen-through content, and not the darkness of the opening. "
            "For transparent or reflective surfaces, estimate the surface finish itself, usually smoother and darker than matte materials, but not automatically zero roughness and not automatically saturated black. "
            "Use broad finish classes such as matte paint, glossy ceramic, glass, polished metal, lacquered wood, stone, and fabric only as weak priors, then refine roughness from the actual visible finish cues in the image. "
            "Typical anchors: polished metal, glossy ceramic, glossy lacquer, clear coated surfaces, mirrors, and smooth glass are usually lower roughness and should appear darker; matte painted walls, plaster, matte wood, fabric, concrete, and diffuse stone are usually higher roughness and should appear brighter unless the image clearly indicates otherwise. "
            "If evidence is weak or ambiguous, prefer conservative but material-aware estimates instead of noise, while still preserving plausible contrast between distinct material classes. "
            "Preserve supported low-amplitude microtexture or repeated surface texture when it is visibly tied to the material finish, but do not invent unsupported fine detail. "
            "Do not stylize the result. "
            "The result should look like a plausible material roughness map, not like a relit grayscale photo, not like a shadow map, not like a foggy constant map, and not like a noisy path-tracing render. "
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
    image_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
    quality: str,
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
            quality=quality,
            output_format="png",
        )


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


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
    image_deployment: str,
    client_kind: str,
    prompt_text: str,
    seg_suffix: str,
    example_rgb: str,
    example_roughness: str,
    timeout: int,
    size: str,
    image_quality: str,
    watermark: bool,
) -> Dict[str, Any]:
    if input_mode == "rgb_only":
        route = "gpt_image_rgb_only"
    elif input_mode == "rgb_plus_example":
        route = "gpt_image_rgb_plus_example"
    else:
        route = "gpt_image_rgb_sam3_soft_direct"

    return {
        "image_model": image_model,
        "image_deployment": image_deployment,
        "client_kind": client_kind,
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
        "image_quality": image_quality,
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


def generate_roughness_map_with_gpt_image(
    *,
    image_client: Any,
    model: str,
    input_mode: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    save_path: Path,
    example_rgb: Optional[Path],
    example_roughness: Optional[Path],
    size: str,
    watermark: bool,
    quality: str,
    timeout: int,
    save_debug_intermediates: bool,
) -> Tuple[str, Dict[str, Any]]:
    preprocess_meta: Dict[str, Any] = {
        "preprocess_canvas_size": f"{DEFAULT_EDIT_CANVAS_SIZE[0]}x{DEFAULT_EDIT_CANVAS_SIZE[1]}",
        "padding_mode": "edge_replicate",
        "crop_reference_size": f"{DEFAULT_EDIT_CANVAS_SIZE[0]}x{DEFAULT_EDIT_CANVAS_SIZE[1]}",
    }

    if input_mode == "rgb_only":
        prompt = build_roughness_prompt(input_mode)
        mode = "gpt_image_rgb_only"
        input_order = ["query_rgb"]
    elif input_mode == "rgb_plus_example":
        if not example_rgb or not example_roughness:
            raise ValueError("rgb_plus_example requires both --example_rgb and --example_roughness")
        prompt = build_roughness_prompt(input_mode)
        mode = "gpt_image_rgb_plus_example"
        input_order = ["example_rgb", "example_roughness", "query_rgb"]
    else:
        if seg_path is None:
            raise ValueError("rgb_plus_seg requires a matching segmentation image")
        prompt = build_roughness_prompt(input_mode)
        mode = "gpt_image_rgb_sam3_soft_direct"
        input_order = ["query_rgb", "segmentation_prior"]

    final_size = get_image_size(rgb_path)
    with tempfile.TemporaryDirectory(prefix="gpt_edit_pad_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        padded_query = temp_dir / "query.png"
        query_bbox = preprocess_image_with_padding(rgb_path, padded_query)
        preprocess_meta["query_content_bbox"] = query_bbox
        preprocess_meta["source_image_size"] = f"{final_size[0]}x{final_size[1]}"
        preprocess_meta["saved_roughness_size"] = f"{final_size[0]}x{final_size[1]}"

        if input_mode == "rgb_only":
            image_paths = [padded_query]
        elif input_mode == "rgb_plus_example":
            if example_rgb is None or example_roughness is None:
                raise ValueError("rgb_plus_example requires both --example_rgb and --example_roughness")
            padded_example_rgb = temp_dir / "example_rgb.png"
            padded_example_roughness = temp_dir / "example_roughness.png"
            example_rgb_bbox = preprocess_image_with_padding(example_rgb, padded_example_rgb)
            example_roughness_bbox = preprocess_image_with_padding(example_roughness, padded_example_roughness)
            preprocess_meta["example_rgb_content_bbox"] = example_rgb_bbox
            preprocess_meta["example_roughness_content_bbox"] = example_roughness_bbox
            image_paths = [padded_example_rgb, padded_example_roughness, padded_query]
        else:
            if seg_path is None:
                raise ValueError("rgb_plus_seg requires a matching segmentation image")
            padded_seg = temp_dir / "segmentation_prior.png"
            seg_bbox = preprocess_image_with_padding(seg_path, padded_seg)
            preprocess_meta["segmentation_prior_content_bbox"] = seg_bbox
            image_paths = [padded_query, padded_seg]

        response = run_image_generation(
            image_client=image_client,
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            size=size,
            watermark=watermark,
            quality=quality,
        )

        raw_bytes = get_image_response_bytes(response.data[0], timeout=timeout)
        if save_debug_intermediates:
            debug_full = save_path.with_name(f"{save_path.stem}_full.png")
            debug_crop = save_path.with_name(f"{save_path.stem}_crop.png")
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
            preprocess_meta["debug_full_output"] = debug_full.name
            preprocess_meta["debug_crop_output"] = debug_crop.name

        save_path.write_bytes(
            postprocess_generated_image_bytes(
                raw_bytes,
                crop_bbox=query_bbox,
                crop_reference_size=DEFAULT_EDIT_CANVAS_SIZE,
                final_size=final_size,
            )
        )

    enforce_grayscale_png(save_path)
    preprocess_meta["input_order"] = input_order
    preprocess_meta["save_debug_intermediates"] = bool(save_debug_intermediates)
    return mode, preprocess_meta


def main() -> None:
    args = parse_args()
    validate_parts(args.num_parts, args.part_index)
    image_client, image_deployment, client_kind, endpoint, api_version = resolve_image_client_config(args)

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
        image_deployment=image_deployment,
        client_kind=client_kind,
        prompt_text=prompt_text,
        seg_suffix=normalize_seg_suffix(args.seg_suffix),
        example_rgb=str(example_rgb) if example_rgb else "",
        example_roughness=str(example_roughness) if example_roughness else "",
        timeout=args.timeout,
        size=args.size,
        image_quality=args.image_quality,
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
        f"client_kind={client_kind} | "
        f"endpoint={endpoint or 'default'} | "
        f"api_version={api_version or 'default'} | "
        f"image_quality={args.image_quality}"
    )

    setup = dict(run_signature)
    setup.update(
        {
            "filename_suffix": args.filename_suffix,
            "recursive": args.recursive,
            "preserve_relative_dirs": args.preserve_relative_dirs,
            "num_parts": args.num_parts,
            "part_index": args.part_index,
            "endpoint": endpoint,
            "api_version": api_version,
            "preprocess_canvas_size": f"{DEFAULT_EDIT_CANVAS_SIZE[0]}x{DEFAULT_EDIT_CANVAS_SIZE[1]}",
            "padding_mode": "edge_replicate",
            "save_debug_intermediates": bool(args.save_debug_intermediates),
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
            roughness_mode, preprocess_meta = generate_roughness_map_with_gpt_image(
                image_client=image_client,
                model=image_deployment,
                input_mode=args.input_mode,
                rgb_path=rgb_path,
                seg_path=seg_path,
                save_path=roughness_path,
                example_rgb=example_rgb,
                example_roughness=example_roughness,
                size=args.size,
                watermark=args.watermark,
                quality=args.image_quality,
                timeout=args.timeout,
                save_debug_intermediates=args.save_debug_intermediates,
            )
            item["roughness_mode"] = roughness_mode
            item["status"] = "done"
            item["roughness_output"] = relative_output(roughness_path, output_dir)
            item["preprocess_canvas_size"] = preprocess_meta.get("preprocess_canvas_size")
            item["saved_roughness_size"] = preprocess_meta.get("saved_roughness_size")
            output_meta = {
                "image_name": rgb_path.name,
                "relative_image_path": relative_image_path,
                "roughness_output": relative_output(roughness_path, output_dir),
                "input_mode": args.input_mode,
                "run_signature": run_signature,
                "num_parts": args.num_parts,
                "part_index": args.part_index,
                "roughness_mode": roughness_mode,
                **preprocess_meta,
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
