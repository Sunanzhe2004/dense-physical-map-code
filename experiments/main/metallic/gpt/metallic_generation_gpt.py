#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Metallic generation script using GPT Image 1.5 with letterbox preprocessing.

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import shutil
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


DEFAULT_AZURE_ENDPOINT = "https://your-azure-openai-resource.openai.azure.com/"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_BASE_URL = DEFAULT_AZURE_ENDPOINT
DEFAULT_IMAGE_MODEL = "gpt-image-1.5"
DEFAULT_IMAGE_SIZE = "1536x1024"
DEFAULT_IMAGE_QUALITY = "medium"
DEFAULT_TIMEOUT = 1800
SOURCE_IMAGE_SIZE = (1280, 720)
RESIZED_CONTENT_SIZE = (1536, 864)
MODEL_INPUT_SIZE = (1536, 1024)
PROMPT_FAMILY = "metallic_rgb_prompt"
INPUT_MODE_CHOICES = ("rgb_only", "rgb_plus_prompt", "rgb_plus_example")
PROMPT_PRESET_CHOICES = ("v3_visualprior_noboundary",)
SUPPORTED_GPT_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metallic maps with GPT Image 1.5 using the final gate_black prompt."
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
        "--prompt_preset",
        type=str,
        default="v3_visualprior_noboundary",
        choices=PROMPT_PRESET_CHOICES,
        help="Built-in metallic prompt preset. This final script fixes the prompt to the gate_black wording.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="",
        help="Per-image prompt text directory. Required only when --input_mode=rgb_plus_prompt.",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="_prompt.txt",
        help="Prompt suffix matched by RGB stem, e.g. Image_0_0_0001_0.png -> Image_0_0_0001_0_prompt.txt",
    )
    parser.add_argument("--example_rgb", type=str, default="", help="Reference RGB path for rgb_plus_example")
    parser.add_argument(
        "--example_metallic",
        type=str,
        default="",
        help="Reference metallic path for rgb_plus_example",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--azure_endpoint",
        "--base_url",
        dest="azure_endpoint",
        type=str,
        default=os.environ.get("AZURE_METALLIC_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or DEFAULT_AZURE_ENDPOINT,
        help="Azure OpenAI endpoint; --base_url is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=os.environ.get("AZURE_METALLIC_OPENAI_API_VERSION")
        or os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or DEFAULT_API_VERSION,
        help="Azure OpenAI API version.",
    )
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument("--image_deployment", type=str, default=None, help="Azure deployment name, defaults to --image_model")
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="edit",
        choices=["edit"],
        help="GPT Image call path follows roughness_generation_gpt.py and uses images.edit.",
    )
    parser.add_argument(
        "--generate_requires_image",
        action="store_true",
        help="In generate mode, fail if the SDK cannot send the image, instead of silently falling back to text-only generation.",
    )
    parser.add_argument("--size", type=str, default=DEFAULT_IMAGE_SIZE, help="Output size for GPT Image 1.5")
    parser.add_argument("--quality", type=str, default=DEFAULT_IMAGE_QUALITY, choices=["low", "medium", "high", "auto"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--watermark", action="store_true", help="Kept for compatibility; ignored by GPT Image 1.5.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist")
    parser.add_argument(
        "--save_debug_intermediates",
        action="store_true",
        help="Save original input, padded input, raw model output, and restored output for alignment debugging.",
    )
    return parser.parse_args()


def ensure_api_key(cli_api_key: Optional[str] = None) -> str:
    api_key = (
        cli_api_key
        or os.environ.get("AZURE_METALLIC_OPENAI_API_KEY")
        or os.environ.get("AZURE_GPT_IMAGE_15_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing authentication: please provide --api_key or set "
            "AZURE_METALLIC_OPENAI_API_KEY / AZURE_GPT_IMAGE_15_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY."
        )
    return api_key.strip()


def resolve_image_client_config(args: argparse.Namespace) -> Tuple[Any, str, str, Optional[str], Optional[str]]:
    api_key = ensure_api_key(args.api_key)
    endpoint = (
        args.azure_endpoint
        or os.environ.get("AZURE_METALLIC_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_GPT_IMAGE_15_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    api_version = (
        args.api_version
        or os.environ.get("AZURE_METALLIC_OPENAI_API_VERSION")
        or os.environ.get("AZURE_GPT_IMAGE_15_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
    )
    deployment = args.image_deployment or args.image_model
    if endpoint:
        if AzureOpenAI is None:
            raise ImportError('Failed to import openai. Please install it with: pip install openai') from _OPENAI_IMPORT_ERROR
        client = AzureOpenAI(
            api_version=api_version or DEFAULT_API_VERSION,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        return client, deployment, "azure", endpoint, api_version or DEFAULT_API_VERSION
    if OpenAI is None:
        raise ImportError('Failed to import openai. Please install it with: pip install openai') from _OPENAI_IMPORT_ERROR
    return OpenAI(api_key=api_key), deployment, "openai", endpoint, api_version


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


def normalize_gpt_image_size(size: Optional[str]) -> str:
    normalized = (size or DEFAULT_IMAGE_SIZE).strip().lower()
    if normalized not in SUPPORTED_GPT_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported --size: {size}. Expected one of: {', '.join(sorted(SUPPORTED_GPT_IMAGE_SIZES))}"
        )
    return normalized


def parse_size_tuple(size: str) -> Tuple[int, int]:
    normalized = normalize_gpt_image_size(size)
    if normalized == "auto":
        return MODEL_INPUT_SIZE
    width_text, height_text = normalized.split("x", 1)
    return int(width_text), int(height_text)


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


def prepare_image_with_padding(
    image_path: Path,
    *,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> Tuple[Path, Dict[str, Any]]:
    target_w, target_h = target_size
    with Image.open(image_path) as src:
        src_rgb = src.convert("RGB")
        orig_w, orig_h = src_rgb.size
        if (orig_w, orig_h) != SOURCE_IMAGE_SIZE:
            raise ValueError(
                f"Expected input image size {SOURCE_IMAGE_SIZE}, got {(orig_w, orig_h)} for {image_path.as_posix()}"
            )
        resized_w, resized_h = RESIZED_CONTENT_SIZE
        resized = src_rgb.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h))
        left = (target_w - resized_w) // 2
        top = (target_h - resized_h) // 2
        canvas.paste(resized, (left, top))
        _edge_fill_canvas(canvas, resized, left, top)

    temp_dir = Path(tempfile.mkdtemp(prefix="metallic_gpt_input_"))
    temp_path = temp_dir / f"{image_path.stem}_padded.png"
    canvas.save(temp_path)
    meta = {
        "original_size": [orig_w, orig_h],
        "canvas_size": [target_w, target_h],
        "content_box": [left, top, left + resized_w, top + resized_h],
        "resized_size": [resized_w, resized_h],
        "temp_path": str(temp_path),
    }
    return temp_path, meta


def scale_bbox_to_image(
    bbox: List[int],
    reference_size: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox length must be 4, got: {bbox}")
    ref_w, ref_h = reference_size
    img_w, img_h = image_size
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
    preprocess_meta: Dict[str, Any],
    final_size: Optional[Tuple[int, int]] = None,
) -> bytes:
    canvas_w, canvas_h = preprocess_meta["canvas_size"]
    left, top, right, bottom = preprocess_meta["content_box"]
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        crop_box = scale_bbox_to_image([left, top, right, bottom], (canvas_w, canvas_h), img.size)
        img = img.crop(crop_box)
        if final_size is not None:
            img = img.resize(final_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        img.convert("L").save(output, format="PNG")
        return output.getvalue()


def list_images(input_dir: Path) -> List[Path]:
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".png"
    ]
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


def build_metallic_prompt(
    input_mode: str,
    prompt_preset: str = "v3_visualprior_noboundary",
    scene_prompt: str = "",
) -> str:
    preset = str(prompt_preset or "v3_visualprior_noboundary").strip().lower()
    if preset != "v3_visualprior_noboundary":
        raise ValueError(f"Unsupported prompt_preset: {prompt_preset}")

    texture_guidance = (
        " As a weak visual prior, bright reflections, mirror-like appearance, or smooth highlights are not sufficient evidence of metal by themselves. "
        " Do not display object boundaries, panel boundaries, contour lines, reflective edges, or decorative lines as gray metallic structure. "
        " Do not use metallic values to preserve boundaries, silhouettes, rims, borders, or edge visibility on likely non-metal regions. "
        " Bright regions caused mainly by direct lighting, light pools, glossy floor reflections, bloom, or illumination falloff are not metal evidence by themselves. "
        " Spatial alignment means pixel correspondence only; do not encode object shape using edges, seams, contour lines, or decorative outlines. "
        " Thin borders, seams, frame lines, grout lines, and decorative contours should remain black unless they belong to a clearly filled exposed-metal surface. "
        " If metal evidence is local or uncertain, keep the rest of the object black rather than extending gray/white into the surrounding housing, frame, or panel. "
        " Outside clearly exposed metal regions, the output should be near-uniform black with no line-like traces or residual edge responses. "
        " When uncertain, do not preserve scene structure with low gray; prefer conservative near-black predictions for likely non-metal regions. "
    )

    if input_mode == "rgb_plus_example":
        prompt = (
            "You are a senior PBR material analysis expert. "
            "You are given exactly three input images in order: "
            "(1) reference RGB image, "
            "(2) reference metallic map, "
            "(3) target RGB image. "
            "Generate only the target metallic map for image (3). "
            "Use image (2) only as output-style guidance for grayscale metallic-map appearance and sparse near-binary material labeling style. "
            "Do not copy the reference scene layout, object placement, or semantic material identity onto the target scene. "
            "The output must be a single-channel grayscale metallic map aligned with the target RGB image. "
            "Preserve the target scene layout exactly. "
            "Do not add, remove, duplicate, move, deform, or hallucinate objects or structures. "
            "Black means dielectric or non-metal. White means clearly exposed metal surface with strong visible metal evidence. "
            "This target convention is effectively binary. Most indoor pixels should be pure black. "
            "Large non-metal regions must remain black with no contour tracing, no dark-gray embossing, and no structural outline rendering. "
            "Do not preserve object boundaries by drawing gray edges around non-metal objects. "
            "Do not output a relit grayscale scene, an edge map, or a reflectance sketch. "
            "Use white only when the visible surface itself provides clear evidence of exposed metal material, such as unmistakable bare metal hardware or an obviously metallic housing surface. "
            "Do not fill an entire object white unless the visible surface is unmistakably an exposed metal body or housing. If metal evidence is local or uncertain, keep the rest black rather than filling the whole object. "
            "Do not treat illumination-driven bright patches, light pools, floor highlights, glossy reflections, or lamp glow as metallic unless the visible surface itself clearly reads as exposed metal. "
            "Spatial alignment means pixel correspondence only; do not encode object shape using edges, seams, contour lines, or decorative outlines. "
            "Thin borders, seams, frame lines, grout lines, and decorative contours should remain black unless they belong to a clearly filled exposed-metal surface. "
            "Mirror frames, window frames, door frames, trims, moldings, cabinet borders, decorative borders, mirror edges, and glossy border strips should remain black unless exposed bare metal evidence is unmistakable. "
            "Large smooth panels, painted appliance sides, coated boards, cabinet faces, drawer fronts, glossy doors, and other broad plain surfaces should remain black unless explicit exposed-metal cues are clearly visible. "
            "Do not convert seams, grooves, panel boundaries, wood grain, tile joints, decorative edges, contour lines, or texture contrast on non-metal materials into gray metallic structure. "
            "Do not use low gray to preserve scene structure, object identity, silhouettes, rims, borders, or reflective outlines. "
            "Glass and mirror remain black. Ceramic, painted walls, stone, wood, plastic, laminate, painted board, coated furniture, and glossy painted surfaces remain black. "
            "For lamps, do not assume the bright diffuser, shade, glowing cover, rim highlight, or reflective ring is metallic; only clearly exposed opaque metal parts may be white. "
            "Small hardware such as knobs, hinges, handles, and brackets may be white only when exposed metal is visually unmistakable; otherwise keep them black with no gray outlining. "
            "If no unmistakable exposed metal surface is visible, output an almost-all-black map. "
            "Outside clearly exposed metal regions, enforce near-uniform black and suppress any residual line-like traces. "
            "Isolated line-like or hollow outline responses should be treated as non-metal unless they belong to a small filled exposed-metal patch with visible area. "
            "If evidence is ambiguous, choose black rather than gray. Mid-gray should be extremely rare. "
            "Do not infer metal from brightness, reflections, shadows, specular highlights, mirror-like appearance, or smoothness alone. "
            f"{texture_guidance}"
            "A good output should look almost empty except for clearly metallic parts. "
            "Prefer a clean material-category map over a photometric grayscale rendering. "
            "Output exactly one grayscale metallic map only."
        )
    else:
        prompt = (
            "You are a senior PBR material analysis expert. "
            "You are given exactly one input image: the target RGB image. "
            "Generate only the target metallic map for that RGB image. "
            "The output must be a single-channel grayscale metallic map spatially aligned with the input RGB content. "
            "Preserve the exact scene layout and object shapes from the RGB image. "
            "Do not add, remove, duplicate, move, deform, or hallucinate objects or structures. "
            "Black means dielectric or non-metal. White means clearly exposed metal surface with strong visible metal evidence. "
            "This target convention is effectively binary. Most indoor pixels should be pure black. "
            "Large non-metal regions must remain black with no contour tracing, no dark-gray embossing, and no structural outline rendering. "
            "Do not preserve object boundaries by drawing gray edges around non-metal objects. "
            "Do not output a relit grayscale scene, an edge map, or a reflectance sketch. "
            "Use white only when the visible surface itself provides clear evidence of exposed metal material, such as unmistakable bare metal hardware or an obviously metallic housing surface. "
            "Do not fill an entire object white unless the visible surface is unmistakably an exposed metal body or housing. If metal evidence is local or uncertain, keep the rest black rather than filling the whole object. "
            "Do not treat illumination-driven bright patches, light pools, floor highlights, glossy reflections, or lamp glow as metallic unless the visible surface itself clearly reads as exposed metal. "
            "Spatial alignment means pixel correspondence only; do not encode object shape using edges, seams, contour lines, or decorative outlines. "
            "Thin borders, seams, frame lines, grout lines, and decorative contours should remain black unless they belong to a clearly filled exposed-metal surface. "
            "Mirror frames, window frames, door frames, trims, moldings, cabinet borders, decorative borders, mirror edges, and glossy border strips should remain black unless exposed bare metal evidence is unmistakable. "
            "Large smooth panels, painted appliance sides, coated boards, cabinet faces, drawer fronts, glossy doors, and other broad plain surfaces should remain black unless explicit exposed-metal cues are clearly visible. "
            "Do not convert seams, grooves, panel boundaries, wood grain, tile joints, decorative edges, contour lines, or texture contrast on non-metal materials into gray metallic structure. "
            "Do not use low gray to preserve scene structure, object identity, silhouettes, rims, borders, or reflective outlines. "
            "Glass and mirror remain black. Ceramic, painted walls, stone, wood, plastic, laminate, painted board, coated furniture, and glossy painted surfaces remain black. "
            "For lamps, do not assume the bright diffuser, shade, glowing cover, rim highlight, or reflective ring is metallic; only clearly exposed opaque metal parts may be white. "
            "Small hardware such as knobs, hinges, handles, and brackets may be white only when exposed metal is visually unmistakable; otherwise keep them black with no gray outlining. "
            "If no unmistakable exposed metal surface is visible, output an almost-all-black map. "
            "Outside clearly exposed metal regions, enforce near-uniform black and suppress any residual line-like traces. "
            "Isolated line-like or hollow outline responses should be treated as non-metal unless they belong to a small filled exposed-metal patch with visible area. "
            "If evidence is ambiguous, choose black rather than gray. Mid-gray should be extremely rare. "
            "Do not infer metal from brightness, reflections, shadows, specular highlights, mirror-like appearance, or smoothness alone. "
            f"{texture_guidance}"
            "A good output should look almost empty except for clearly metallic parts. "
            "Prefer a sparse, semantically correct material-category map over a smooth grayscale rendering. "
            "When evidence is ambiguous, favor conservative non-metal predictions rather than inventing large metallic regions. "
            "Output exactly one grayscale metallic map only."
        )

    if input_mode == "rgb_plus_prompt":
        soft_scene_prompt = scene_prompt.strip()
        if soft_scene_prompt:
            prompt += (
                " Additional material prior is provided below as a weak text prompt. "
                "Use it only when it is consistent with the RGB image, and never let it override visible image evidence. "
                "Treat it as soft guidance about likely metal versus dielectric assignments, not as guaranteed ground truth. "
                f"Prompt prior: {soft_scene_prompt} "
            )

    return prompt

def run_image_generation(
    image_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
    quality: str,
) -> Tuple[Any, str]:
    del watermark
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    requested_size = normalize_gpt_image_size(size)

    with contextlib.ExitStack() as stack:
        image_files = [stack.enter_context(path.open("rb")) for path in image_paths]
        image_arg: Any = image_files[0] if len(image_files) == 1 else image_files
        response = image_client.images.edit(
            model=model,
            image=image_arg,
            prompt=prompt,
            n=1,
            size=requested_size,
            quality=quality,
            output_format="png",
        )
        return response, requested_size


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
    preprocess_meta: Optional[Dict[str, Any]] = None,
    final_size: Optional[Tuple[int, int]] = None,
) -> None:
    image_bytes = get_image_response_bytes(image_item, timeout=timeout)
    if preprocess_meta is not None:
        image_bytes = postprocess_generated_image_bytes(
            image_bytes,
            preprocess_meta=preprocess_meta,
            final_size=final_size,
        )
    save_path.write_bytes(image_bytes)


def copy_image_for_debug(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def save_debug_bundle(
    *,
    rgb_path: Path,
    prepared_target_path: Path,
    raw_image_bytes: bytes,
    final_output_path: Path,
) -> None:
    debug_dir = final_output_path.parent.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = final_output_path.stem
    copy_image_for_debug(rgb_path, debug_dir / f"{stem}_01_original.png")
    copy_image_for_debug(prepared_target_path, debug_dir / f"{stem}_02_padded_input.png")
    (debug_dir / f"{stem}_03_raw_generated.png").write_bytes(raw_image_bytes)
    copy_image_for_debug(final_output_path, debug_dir / f"{stem}_04_restored_output.png")


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
    prompt_version: str,
    prompt_preset: str,
    image_model: str,
    prompt_text: str,
    prompt_dir: str,
    prompt_suffix: str,
    example_rgb: str,
    example_metallic: str,
    timeout: int,
    size: str,
    quality: str,
    seed: int,
    watermark: bool,
) -> Dict[str, Any]:
    if input_mode == "rgb_plus_prompt":
        route = "gpt_image_rgb_prompt_metallic"
    elif input_mode == "rgb_plus_example":
        route = "gpt_image_rgb_example_metallic"
    else:
        route = "gpt_image_rgb_only_metallic"

    return {
        "image_model": image_model,
        "route": route,
        "input_mode": input_mode,
        "prompt_version": prompt_version,
        "prompt_preset": prompt_preset,
        "prompt_text": prompt_text,
        "prompt_dir": prompt_dir,
        "prompt_suffix": prompt_suffix,
        "example_rgb": example_rgb,
        "example_metallic": example_metallic,
        "timeout": timeout,
        "size": size,
        "quality": quality,
        "seed": seed,
        "model_input_size": list(MODEL_INPUT_SIZE),
        "output_resolution_policy": "crop_to_original_after_letterbox",
        "watermark": watermark,
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


def format_exception_with_context(
    exc: Exception,
    *,
    endpoint: Optional[str],
    model: str,
    input_mode: str,
    image_paths: List[Path],
) -> str:
    exc_type = type(exc).__name__
    exc_text = str(exc).strip() or repr(exc)
    image_names = [path.name for path in image_paths]
    parts = [
        f"{exc_type}: {exc_text}",
        f"model={model}",
        f"input_mode={input_mode}",
        f"endpoint={endpoint or 'N/A'}",
        f"images={image_names}",
    ]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        cause_text = str(cause).strip() or repr(cause)
        parts.append(f"cause={type(cause).__name__}: {cause_text}")
    return " | ".join(parts)


def generate_metallic_map_with_gpt_image(
    *,
    client: Any,
    endpoint: Optional[str],
    model: str,
    input_mode: str,
    prompt_preset: str,
    rgb_path: Path,
    save_path: Path,
    scene_prompt: str,
    example_rgb: Optional[Path],
    example_metallic: Optional[Path],
    size: str,
    quality: str,
    seed: int,
    timeout: int,
    save_debug_intermediates: bool,
) -> str:
    prompt = build_metallic_prompt(
        input_mode=input_mode,
        prompt_preset=prompt_preset,
        scene_prompt=scene_prompt,
    )
    prepared_paths: List[Path] = []
    temp_dirs: List[Path] = []
    primary_preprocess_meta: Optional[Dict[str, Any]] = None
    target_prepared_path: Optional[Path] = None
    try:
        if input_mode == "rgb_plus_example":
            if not example_rgb or not example_metallic:
                raise ValueError("rgb_plus_example requires both --example_rgb and --example_metallic")
            for path in (example_rgb, example_metallic, rgb_path):
                prepared_path, meta = prepare_image_with_padding(path)
                prepared_paths.append(prepared_path)
                temp_dirs.append(prepared_path.parent)
                if path == rgb_path:
                    primary_preprocess_meta = meta
                    target_prepared_path = prepared_path
            mode = "gpt_image_rgb_example_metallic"
            response, requested_size = run_image_generation(
                image_client=client,
                model=model,
                prompt=prompt,
                image_paths=prepared_paths,
                size=size,
                watermark=False,
                quality=quality,
            )
        elif input_mode == "rgb_plus_prompt":
            prepared_path, primary_preprocess_meta = prepare_image_with_padding(rgb_path)
            prepared_paths = [prepared_path]
            temp_dirs.append(prepared_path.parent)
            target_prepared_path = prepared_path
            mode = "gpt_image_rgb_prompt_metallic"
            response, requested_size = run_image_generation(
                image_client=client,
                model=model,
                prompt=prompt,
                image_paths=prepared_paths,
                size=size,
                watermark=False,
                quality=quality,
            )
        else:
            prepared_path, primary_preprocess_meta = prepare_image_with_padding(rgb_path)
            prepared_paths = [prepared_path]
            temp_dirs.append(prepared_path.parent)
            target_prepared_path = prepared_path
            mode = "gpt_image_rgb_only_metallic"
            response, requested_size = run_image_generation(
                image_client=client,
                model=model,
                prompt=prompt,
                image_paths=prepared_paths,
                size=size,
                watermark=False,
                quality=quality,
            )
        final_size = tuple(primary_preprocess_meta["original_size"]) if primary_preprocess_meta is not None else None
        raw_image_bytes = get_image_response_bytes(response.data[0], timeout=timeout)
        if primary_preprocess_meta is not None:
            final_image_bytes = postprocess_generated_image_bytes(
                raw_image_bytes,
                preprocess_meta=primary_preprocess_meta,
                final_size=final_size,
            )
        else:
            final_image_bytes = raw_image_bytes
        save_path.write_bytes(final_image_bytes)
        if save_debug_intermediates and target_prepared_path is not None:
            save_debug_bundle(
                rgb_path=rgb_path,
                prepared_target_path=target_prepared_path,
                raw_image_bytes=raw_image_bytes,
                final_output_path=save_path,
            )
        if primary_preprocess_meta is None:
            raise RuntimeError("Missing preprocess metadata for target RGB image.")
        return mode
    except Exception as exc:
        detail_paths = prepared_paths if prepared_paths else [rgb_path]
        raise RuntimeError(
            format_exception_with_context(
                exc,
                endpoint=endpoint,
                model=model,
                input_mode=input_mode,
                image_paths=detail_paths,
            )
        ) from exc
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    client, image_model, client_kind, resolved_endpoint, resolved_api_version = resolve_image_client_config(args)
    prompt_version = f"{PROMPT_FAMILY}_{args.prompt_preset}"

    input_dir = Path(args.input_dir)
    prompt_dir = Path(args.prompt_dir).expanduser() if args.prompt_dir else None
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

    if args.input_mode == "rgb_plus_prompt":
        if prompt_dir is None or not prompt_dir.exists() or not prompt_dir.is_dir():
            raise FileNotFoundError("rgb_plus_prompt requires a valid --prompt_dir")
    elif args.input_mode == "rgb_plus_example":
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
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} metallic maps with model={image_model}, input_mode={args.input_mode}"
    )
    print(
        "      "
        f"client_kind={client_kind} | "
        f"endpoint={resolved_endpoint or 'N/A'} | "
        f"api_version={resolved_api_version or 'N/A'} | "
        f"quality={args.quality} | "
        f"size={normalize_gpt_image_size(args.size)}"
    )
    prompt_text = build_metallic_prompt(
        args.input_mode,
        prompt_preset=args.prompt_preset,
        scene_prompt="[per-image prompt inserted at runtime]" if args.input_mode == "rgb_plus_prompt" else "",
    )
    run_signature = build_run_signature(
        input_mode=args.input_mode,
        prompt_version=prompt_version,
        prompt_preset=args.prompt_preset,
        image_model=image_model,
        prompt_text=prompt_text,
        prompt_dir=str(prompt_dir) if prompt_dir else "",
        prompt_suffix=args.prompt_suffix,
        example_rgb=str(example_rgb) if example_rgb else "",
        example_metallic=str(example_metallic) if example_metallic else "",
        timeout=args.timeout,
        size=args.size,
        quality=args.quality,
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
            prompt_path: Optional[Path] = None
            scene_prompt = ""
            if args.input_mode == "rgb_plus_prompt":
                if prompt_dir is None:
                    raise FileNotFoundError("rgb_plus_prompt requires --prompt_dir")
                prompt_path = find_matching_prompt(rgb_path, prompt_dir, args.prompt_suffix)
                scene_prompt = load_optional_text(prompt_path)
                if not scene_prompt:
                    raise RuntimeError(f"Prompt text is empty: {prompt_path}")
                item["prompt_name"] = prompt_path.name
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name} | prompt={prompt_path.name}")
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
            item["prompt_version"] = prompt_version
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
                item["metallic_mode"] = generate_metallic_map_with_gpt_image(
                    client=client,
                    endpoint=resolved_endpoint,
                    model=image_model,
                    input_mode=args.input_mode,
                    prompt_preset=args.prompt_preset,
                    rgb_path=rgb_path,
                    save_path=metallic_path,
                    scene_prompt=scene_prompt,
                    example_rgb=example_rgb,
                    example_metallic=example_metallic,
                    size=args.size,
                    quality=args.quality,
                    seed=args.seed,
                    timeout=args.timeout,
                    save_debug_intermediates=args.save_debug_intermediates,
                )
                item["metallic_output"] = metallic_path.name
                output_meta = {
                    "image_name": rgb_path.name,
                    "metallic_output": metallic_path.name,
                    "input_mode": args.input_mode,
                    "run_signature": run_signature,
                }
                if prompt_path is not None:
                    output_meta["prompt_name"] = prompt_path.name
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
