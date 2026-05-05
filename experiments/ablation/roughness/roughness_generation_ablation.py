#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Ablation-capable roughness generation module for Doubao.
# It defines the shared A0/A1/A2/A3 experiment variants and exposes reusable
# helpers for the ablation runner and for direct single-script execution.

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
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
SEG_CANDIDATE_SUFFIXES = ["_seg", "_mask", "_sam", "_semantic", "_label", ""]
INPUT_MODE_CHOICES = ("rgb_plus_seg", "rgb_only", "rgb_plus_example")


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    input_mode: str
    route: str
    segmentation_role: str
    use_example_pair: bool
    postprocess_mode: str
    prompt_version: str


VARIANT_CONFIGS: Dict[str, VariantConfig] = {
    "a0": VariantConfig(
        variant_id="a0",
        variant_name="A0 RGB Only",
        input_mode="rgb_only",
        route="doubao_roughness_a0_rgb_only",
        segmentation_role="unused",
        use_example_pair=False,
        postprocess_mode="none",
        prompt_version="roughness_a0_rgb_only_v1",
    ),
    "a1": VariantConfig(
        variant_id="a1",
        variant_name="A1 Soft Segmentation Prior",
        input_mode="rgb_plus_seg",
        route="doubao_roughness_a1_soft_seg",
        segmentation_role="soft_spatial_prior",
        use_example_pair=False,
        postprocess_mode="none",
        prompt_version="roughness_a1_soft_seg_v1",
    ),
    "a2": VariantConfig(
        variant_id="a2",
        variant_name="A2 Segmentation Region Fill",
        input_mode="rgb_plus_seg",
        route="doubao_roughness_a2_region_fill",
        segmentation_role="hard_region_fill_diagnostic",
        use_example_pair=False,
        postprocess_mode="connected_component_region_mean_fill",
        prompt_version="roughness_a2_region_fill_v2",
    ),
    "a3": VariantConfig(
        variant_id="a3",
        variant_name="A3 Fixed Exemplar Pair",
        input_mode="rgb_plus_example",
        route="doubao_roughness_a3_fixed_exemplar",
        segmentation_role="unused",
        use_example_pair=True,
        postprocess_mode="none",
        prompt_version="roughness_a3_fixed_exemplar_v2",
    ),
}

DEFAULT_VARIANT_BY_INPUT_MODE = {
    "rgb_only": "a0",
    "rgb_plus_seg": "a1",
    "rgb_plus_example": "a3",
}


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
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Ark base URL")
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument("--size", type=str, default="adaptive", help="Output size; adaptive -> 2k")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist")
    return parser.parse_args()


def resolve_default_example_pair(args: argparse.Namespace, script_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    examples_dir = script_dir / "examples"
    example_rgb = Path(args.example_rgb).expanduser() if args.example_rgb else examples_dir / "image.png"
    example_roughness = (
        Path(args.example_roughness).expanduser() if args.example_roughness else examples_dir / "roughness.png"
    )
    return example_rgb, example_roughness


def get_variant_config(variant_id: str) -> VariantConfig:
    key = str(variant_id or "").strip().lower()
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"Unsupported variant_id: {variant_id}")
    return VARIANT_CONFIGS[key]


def get_default_variant_for_input_mode(input_mode: str) -> VariantConfig:
    key = str(input_mode or "").strip().lower()
    if key not in DEFAULT_VARIANT_BY_INPUT_MODE:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return get_variant_config(DEFAULT_VARIANT_BY_INPUT_MODE[key])


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


def list_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"RGB image directory not found: {input_dir}")

    preferred = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.name.lower().endswith("_im.png")
    ]
    if preferred:
        return preferred

    recursive_preferred = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.name.lower().endswith("_im.png")
    ]
    if recursive_preferred:
        return recursive_preferred

    image_prefix = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if image_prefix:
        return image_prefix

    recursive_image_prefix = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() == ".png" and p.name.lower().startswith("image_")
    ]
    if recursive_image_prefix:
        return recursive_image_prefix

    raise FileNotFoundError(f"No RGB images matching *_im.png or Image_*.png were found in {input_dir} or its subdirectories")


def filter_images_by_names(image_paths: Sequence[Path], include_names: Optional[Sequence[str]]) -> List[Path]:
    if not include_names:
        return list(image_paths)

    wanted = [str(name).strip() for name in include_names if str(name).strip()]
    if not wanted:
        return list(image_paths)

    image_map = {path.name: path for path in image_paths}
    image_map.update({path.as_posix(): path for path in image_paths})
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
    candidate_dirs = [seg_dir]

    if rgb_path.parent != seg_dir:
        candidate_dirs.append(rgb_path.parent)

    try:
        rel_parent = rgb_path.parent.relative_to(seg_dir)
    except ValueError:
        rel_parent = None
    if rel_parent is not None and rel_parent != Path("."):
        candidate_dirs.append(seg_dir / rel_parent)

    for candidate_dir in candidate_dirs:
        for suffix in seg_suffixes:
            for ext in [rgb_path.suffix, ".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                candidates.append(candidate_dir / f"{stem}{suffix}{ext}")

    name_lower = rgb_path.name.lower()
    if name_lower.startswith("image_"):
        seg_name = "ObjectSegmentation_" + rgb_path.name[len("Image_"):]
        for candidate_dir in candidate_dirs:
            candidates.append(candidate_dir / seg_name)

    for tail in ["_im", "_rgb", "_image"]:
        if stem.endswith(tail):
            root = stem[: -len(tail)]
            for candidate_dir in candidate_dirs:
                for suffix in seg_suffixes:
                    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                        candidates.append(candidate_dir / f"{root}{suffix}{ext}")

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

    raise FileNotFoundError(f"Cannot find segmentation image for {rgb_path.as_posix()} in {seg_dir}")


def build_prompt(variant_id: str) -> str:
    config = get_variant_config(variant_id)
    if config.input_mode == "rgb_only":
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

    if config.input_mode == "rgb_plus_example":
        return (
            "You are a senior PBR material analysis expert. "
            "You are given exactly three input images in order: "
            "(1) reference RGB image, "
            "(2) reference roughness map, "
            "(3) target RGB image. "
            "Generate only the target roughness map for image (3). "
            "The reference RGB image and reference roughness map form one fixed exemplar pair that is shared across all target images in this experiment. "
            "Use the reference pair only as a global output-style prior for roughness-map appearance, tonal distribution, and plausible grayscale formatting. "
            "The role of the reference RGB image is only to provide visual context for interpreting the reference roughness map; it is not a template for the target scene. "
            "Do not copy the reference scene layout, object arrangement, object boundaries, or semantic material identity onto the target scene. "
            "When the target RGB evidence conflicts with the reference pair, always trust the target RGB evidence for material roughness. "
            "The output must be a single-channel grayscale roughness map spatially aligned with the target RGB content. "
            "Black means very smooth or polished. White means very rough and diffuse. "
            "Roughness ranges: 0.0-0.1 means mirror-like or highly polished, 0.1-0.3 means smooth or glossy, 0.3-0.6 means semi-gloss to moderately rough, 0.6-0.85 means rough diffuse material, and 0.85-1.0 means very rough, heavily scattering surfaces. "
            "Output one image only: no color, no text, no labels, no collage, and no overlay. "
            "Do not copy RGB brightness, direct illumination, cast shadows, self-shadowing, ambient occlusion, reflections, or bright highlights into roughness. "
            "Do not convert low-frequency lighting gradients into low-frequency roughness gradients. "
            "If the same material appears under different lighting, keep the roughness similar across those pixels even if the RGB brightness changes strongly. "
            "Infer roughness from material appearance, highlight sharpness, reflection behavior, coating cues, and visible surface micro-structure cues, not from brightness alone. "
            "Use the reference pair as output-style guidance only, not as a material template, semantic prior, or layout prior for the target. "
            "Prefer a piecewise-smooth material map over a photometric grayscale rendering. "
            "Large homogeneous surfaces should usually remain spatially stable when the RGB evidence supports that. "
            "Albedo patterns, color changes, printed textures, and semantic identity do not automatically imply strong roughness variation. "
            "Local roughness variation should appear only when the visible material micro-structure clearly supports it; shading bands and lighting gradients must not dominate the map. "
            "When uncertain, prefer conservative and spatially stable roughness estimates rather than exaggerated local contrast. "
            "Do not stylize the result. Do not invent high-frequency texture where the RGB image does not support it. "
            "Output exactly one grayscale roughness map only."
        )

    if config.variant_id == "a1":
        return (
            "You are a senior PBR material analysis expert. "
            "You are given exactly two input images in order: "
            "(1) target RGB image, "
            "(2) target segmentation prior map. "
            "Generate only the target roughness map for the RGB image. "
            "Use the segmentation map as a soft spatial prior for object and material boundaries, not as guaranteed material segmentation. "
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

    if config.variant_id == "a2":
        return (
            "You are a senior PBR material analysis expert. "
            "You are given exactly two input images in order: "
            "(1) target RGB image, "
            "(2) target semantic segmentation map. "
            "Generate only the target roughness map. "
            "Segmentation has very high priority: pixels within the same segmented region or connected coherent subregion should usually remain within a consistent roughness range unless clear RGB evidence indicates a real material change. "
            "Treat segmentation as a strong region-level constraint for spatial consistency, connected-region coherence, and boundary placement. "
            "The output must be a single-channel grayscale roughness map spatially aligned with the target RGB content. "
            "Black means very smooth or polished. White means very rough and diffuse. "
            "Roughness ranges: 0.0-0.1 means mirror-like or highly polished, 0.1-0.3 means smooth or glossy, 0.3-0.6 means semi-gloss to moderately rough, 0.6-0.85 means rough diffuse material, and 0.85-1.0 means very rough, heavily scattering surfaces. "
            "Output one image only: no color, no text, no labels, no collage, and no overlay. "
            "Do not trace illumination, cast shadows, self-shadowing, bright highlights, reflections, exposure, or ambient occlusion directly as roughness. "
            "Use visual evidence such as highlight sharpness, reflection behavior, texture density, coating cues, and material appearance, but do not convert RGB brightness into roughness. "
            "Do not systematically underestimate roughness. "
            "Use dark low-roughness tones only when the region shows clear smooth, polished, coated, reflective, or sharp-specular cues. "
            "If a region lacks strong gloss evidence, prefer a moderate or moderately high roughness range over a smooth-looking dark fill. "
            "Large planar painted or coated parts should usually have stable roughness values within the same segmented region. "
            "Fabric-like parts may have moderate local variation, but folds, shading bands, and lighting gradients must not dominate the roughness map. "
            "Keep object boundaries and material boundaries sharp where supported by the segmentation map. "
            "Prefer region-level stability over fine photometric detail. "
            "Maintain a plausible tonal spread across regions: clearly diffuse regions should usually be noticeably brighter than clearly smooth glossy regions. "
            "If evidence inside one segmented part is ambiguous, keep the roughness range spatially consistent rather than introducing unnecessary within-region variation. "
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


def build_roughness_prompt(input_mode: str) -> str:
    config = get_default_variant_for_input_mode(input_mode)
    return build_prompt(config.variant_id)


def resolve_input_images(
    variant_id: str,
    rgb_path: Path,
    seg_path: Optional[Path] = None,
    example_pair: Optional[Tuple[Path, Path]] = None,
) -> List[Path]:
    config = get_variant_config(variant_id)
    if config.input_mode == "rgb_only":
        return [rgb_path]
    if config.input_mode == "rgb_plus_example":
        if example_pair is None:
            raise ValueError(f"{variant_id} requires an example_pair")
        example_rgb, example_roughness = example_pair
        return [example_rgb, example_roughness, rgb_path]
    if seg_path is None:
        raise ValueError(f"{variant_id} requires a matching segmentation image")
    return [rgb_path, seg_path]


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
    with Image.open(path) as image:
        image.convert("L").save(path)


def apply_region_fill_from_segmentation(pred_path: Path, seg_path: Path, save_path: Path) -> None:
    with Image.open(pred_path) as pred_img:
        pred_np = np.array(pred_img.convert("L"), dtype=np.float32)

    with Image.open(seg_path) as seg_img:
        seg_rgb_img = seg_img.convert("RGB")
        pred_height, pred_width = pred_np.shape[:2]
        if seg_rgb_img.size != (pred_width, pred_height):
            seg_rgb_img = seg_rgb_img.resize((pred_width, pred_height), Image.Resampling.NEAREST)
        seg_rgb = np.array(seg_rgb_img, dtype=np.uint32)

    label_map = (seg_rgb[..., 0] << 16) | (seg_rgb[..., 1] << 8) | seg_rgb[..., 2]
    height, width = label_map.shape
    visited = np.zeros((height, width), dtype=bool)
    output = pred_np.copy()

    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            label_value = int(label_map[y, x])
            queue: deque[Tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component_pixels: List[Tuple[int, int]] = []

            while queue:
                cy, cx = queue.popleft()
                component_pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or int(label_map[ny, nx]) != label_value:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

            values = [pred_np[py, px] for py, px in component_pixels]
            mean_value = float(np.mean(values)) if values else float(pred_np[y, x])
            for py, px in component_pixels:
                output[py, px] = mean_value

    Image.fromarray(np.clip(np.round(output), 0, 255).astype(np.uint8), mode="L").save(save_path)


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
    image_key = str(item.get("image_relpath") or item.get("image_name") or "").strip()
    if not image_key:
        manifest.append(item)
        return
    for idx, existing in enumerate(manifest):
        existing_key = str(existing.get("image_relpath") or existing.get("image_name") or "").strip()
        if existing_key == image_key:
            manifest[idx] = item
            return
    manifest.append(item)


def build_run_signature(
    *,
    variant_id: str,
    image_model: str,
    prompt_text: str,
    seg_suffix: str,
    example_rgb: str,
    example_roughness: str,
    timeout: int,
    size: str,
    watermark: bool,
) -> Dict[str, Any]:
    config = get_variant_config(variant_id)
    segmentation_assumption = "unused"
    if config.variant_id == "a1":
        segmentation_assumption = "not_guaranteed_material_segmentation"
    elif config.variant_id == "a2":
        segmentation_assumption = "connected_component_region_mean_fill_diagnostic"

    example_pair_policy = "unused"
    example_rgb_role = "unused"
    if config.variant_id == "a3":
        example_pair_policy = "fixed_global_exemplar_pair"
        example_rgb_role = "reference_context_for_reference_roughness"

    return {
        "variant_id": config.variant_id,
        "variant_name": config.variant_name,
        "image_model": image_model,
        "route": config.route,
        "input_mode": config.input_mode,
        "prompt_version": config.prompt_version,
        "prompt_text": prompt_text,
        "segmentation_role": config.segmentation_role,
        "segmentation_assumption": segmentation_assumption,
        "postprocess_mode": config.postprocess_mode,
        "seg_suffix": seg_suffix,
        "example_rgb": example_rgb,
        "example_roughness": example_roughness,
        "example_pair_policy": example_pair_policy,
        "example_rgb_role": example_rgb_role,
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
    if not roughness_path.exists() or not output_meta_path.exists():
        return False
    saved = load_json_dict(output_meta_path)
    return saved.get("run_signature") == run_signature


def generate_one_case(
    *,
    ark_client: Any,
    model: str,
    variant_id: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    save_path: Path,
    example_rgb: Optional[Path],
    example_roughness: Optional[Path],
    size: str,
    watermark: bool,
    timeout: int,
    source_prediction_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = get_variant_config(variant_id)
    example_pair = None
    if config.use_example_pair:
        if example_rgb is None or example_roughness is None:
            raise ValueError(f"{variant_id} requires both example_rgb and example_roughness")
        example_pair = (example_rgb, example_roughness)

    prompt = build_prompt(config.variant_id)
    image_paths = resolve_input_images(
        config.variant_id,
        rgb_path=rgb_path,
        seg_path=seg_path,
        example_pair=example_pair,
    )

    source_prediction = None
    if config.variant_id == "a2":
        if seg_path is None:
            raise ValueError("a2 requires a matching segmentation image")
        source_prediction = source_prediction_path or save_path.with_name(f"{save_path.stem}_source.png")
        response = run_image_generation(
            ark_client=ark_client,
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            size=size,
            watermark=watermark,
        )
        save_image_response(response.data[0], source_prediction, timeout=timeout)
        enforce_grayscale_png(source_prediction)
        apply_region_fill_from_segmentation(source_prediction, seg_path, save_path)
    else:
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

    result: Dict[str, Any] = {
        "variant_id": config.variant_id,
        "variant_name": config.variant_name,
        "route": config.route,
        "input_mode": config.input_mode,
        "prompt_text": prompt,
        "segmentation_role": config.segmentation_role,
        "postprocess_mode": config.postprocess_mode,
        "example_rgb": str(example_rgb) if example_rgb else "",
        "example_roughness": str(example_roughness) if example_roughness else "",
    }
    if source_prediction is not None:
        result["source_prediction"] = source_prediction.name
    return result


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
    config = get_default_variant_for_input_mode(input_mode)
    result = generate_one_case(
        ark_client=ark_client,
        model=model,
        variant_id=config.variant_id,
        rgb_path=rgb_path,
        seg_path=seg_path,
        save_path=save_path,
        example_rgb=example_rgb,
        example_roughness=example_roughness,
        size=size,
        watermark=watermark,
        timeout=timeout,
    )
    return str(result["route"])


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key()

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

    default_variant = get_default_variant_for_input_mode(args.input_mode)
    input_dir = Path(args.input_dir)
    seg_dir = Path(args.seg_dir).expanduser() if args.seg_dir else None
    script_dir = Path(__file__).resolve().parent
    example_rgb = None
    example_roughness = None

    if args.input_mode == "rgb_plus_example":
        example_rgb, example_roughness = resolve_default_example_pair(args, script_dir)
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

    image_paths = list_images(input_dir)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths

    print(
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} roughness maps with model={args.image_model}, input_mode={args.input_mode}"
    )

    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    prompt_text = build_prompt(default_variant.variant_id)
    run_signature = build_run_signature(
        variant_id=default_variant.variant_id,
        image_model=args.image_model,
        prompt_text=prompt_text,
        seg_suffix=normalize_seg_suffix(args.seg_suffix),
        example_rgb=str(example_rgb) if example_rgb else "",
        example_roughness=str(example_roughness) if example_roughness else "",
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
        try:
            image_relpath = rgb_path.relative_to(input_dir)
        except ValueError:
            image_relpath = Path(rgb_path.name)
        roughness_relpath = image_relpath.with_name(f"{image_relpath.stem}_roughness.png")

        item: Dict[str, Any] = {
            "image_name": rgb_path.name,
            "image_relpath": image_relpath.as_posix(),
        }
        try:
            seg_path: Optional[Path] = None
            if args.input_mode == "rgb_plus_seg":
                if seg_dir is None:
                    raise ValueError("rgb_plus_seg requires --seg_dir")
                seg_path = find_matching_seg(rgb_path, seg_dir, preferred_suffix=args.seg_suffix)
                item["seg_name"] = seg_path.name
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name} | seg={seg_path.name}")
            else:
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}")

            roughness_path = roughness_dir / roughness_relpath
            output_meta_path = output_meta_dir / roughness_relpath.with_suffix(".json")
            roughness_path.parent.mkdir(parents=True, exist_ok=True)
            output_meta_path.parent.mkdir(parents=True, exist_ok=True)
            item["prompt_version"] = default_variant.prompt_version
            item["input_mode"] = args.input_mode
            item["variant_id"] = default_variant.variant_id
            if args.skip_existing and should_skip_existing_output(
                roughness_path=roughness_path,
                output_meta_path=output_meta_path,
                run_signature=run_signature,
            ):
                item["skipped"] = True
                item["skip_reason"] = "matching_output_and_signature"
                item["roughness_output"] = roughness_relpath.as_posix()
            else:
                generated = generate_one_case(
                    ark_client=ark_client,
                    model=args.image_model,
                    variant_id=default_variant.variant_id,
                    rgb_path=rgb_path,
                    seg_path=seg_path,
                    save_path=roughness_path,
                    example_rgb=example_rgb,
                    example_roughness=example_roughness,
                    size=args.size,
                    watermark=args.watermark,
                    timeout=args.timeout,
                )
                item["roughness_mode"] = generated["route"]
                item["roughness_output"] = roughness_relpath.as_posix()
                output_meta: Dict[str, Any] = {
                    "image_name": rgb_path.name,
                    "image_relpath": image_relpath.as_posix(),
                    "roughness_output": roughness_relpath.as_posix(),
                    "variant_id": default_variant.variant_id,
                    "variant_name": default_variant.variant_name,
                    "input_mode": args.input_mode,
                    "run_signature": run_signature,
                }
                if seg_path is not None:
                    output_meta["seg_name"] = seg_path.name
                if generated.get("source_prediction"):
                    output_meta["source_prediction"] = generated["source_prediction"]
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
