#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final ablation-capable metallic generation module for Doubao/SeedDream.

Design:
  A0 = main RGB metallic prompt; target RGB only; no per-image scene prompt.
  A1 = A0 + soft oracle object/instance segmentation prior image.
  A2 = A0 + strong oracle segmentation prior + connected-component mean-fill diagnostic.
  A3 = A0 + fixed RGB/metallic exemplar pair.

This file intentionally reuses the final main metallic script shipped in the
repository, so A0 stays comparable with the main experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _load_main_metallic_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "experiments" / "main" / "metallic" / "doubao" / "metallic_generation_doubao_final.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Main Doubao metallic script not found: {module_path}")

    spec = importlib.util.spec_from_file_location("metallic_generation_doubao_final", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MAIN_METALLIC = _load_main_metallic_module()

Ark = _MAIN_METALLIC.Ark
DEFAULT_BASE_URL = _MAIN_METALLIC.DEFAULT_BASE_URL
DEFAULT_IMAGE_MODEL = _MAIN_METALLIC.DEFAULT_IMAGE_MODEL
DEFAULT_TIMEOUT = _MAIN_METALLIC.DEFAULT_TIMEOUT
_ARK_IMPORT_ERROR = _MAIN_METALLIC._ARK_IMPORT_ERROR
build_metallic_prompt = _MAIN_METALLIC.build_metallic_prompt
enforce_grayscale_png = _MAIN_METALLIC.enforce_grayscale_png
ensure_api_key = _MAIN_METALLIC.ensure_api_key
infer_rgb_dir_from_seg_dir = _MAIN_METALLIC.infer_rgb_dir_from_seg_dir
list_images = _MAIN_METALLIC.list_images
load_json_dict = _MAIN_METALLIC.load_json_dict
load_manifest = _MAIN_METALLIC.load_manifest
looks_like_object_seg_dir = _MAIN_METALLIC.looks_like_object_seg_dir
run_image_generation = _MAIN_METALLIC.run_image_generation
save_image_response = _MAIN_METALLIC.save_image_response
upsert_manifest_entry = _MAIN_METALLIC.upsert_manifest_entry
write_manifest = _MAIN_METALLIC.write_manifest

PROMPT_PRESET = "v3_visualprior_noboundary"
SEG_CANDIDATE_SUFFIXES = ["_seg", "_mask", "_sam", "_semantic", "_label", ""]


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    description: str
    input_mode: str
    route: str
    prompt_version: str
    segmentation_role: str = "unused"
    example_pair_policy: str = "unused"
    postprocess_mode: str = "none"
    use_segmentation: bool = False
    use_example_pair: bool = False


VARIANT_CONFIGS: Dict[str, VariantConfig] = {
    "a0": VariantConfig(
        variant_id="a0",
        variant_name="A0 Main RGB Prompt",
        description="Main experiment protocol: target RGB only with the shared main metallic prompt.",
        input_mode="rgb_only",
        route="seedream_metallic_a0_rgb_main",
        prompt_version="metallic_a0_main_rgb_prompt_v3_visualprior_noboundary",
    ),
    "a1": VariantConfig(
        variant_id="a1",
        variant_name="A1 Main + Soft Segmentation Prior",
        description="Target RGB + soft object/instance segmentation prior, sharing the A0 metallic core.",
        input_mode="rgb_plus_seg",
        route="seedream_metallic_a1_soft_seg",
        prompt_version="metallic_a1_soft_seg_v3_shared_core",
        segmentation_role="soft_object_instance_spatial_prior",
        use_segmentation=True,
    ),
    "a2": VariantConfig(
        variant_id="a2",
        variant_name="A2 Main + Segmentation Region Fill",
        description="Target RGB + strong segmentation prior + connected-component mean-fill diagnostic, sharing the A0 metallic core.",
        input_mode="rgb_plus_seg",
        route="seedream_metallic_a2_region_fill",
        prompt_version="metallic_a2_region_fill_v4_strict_shared_core",
        segmentation_role="strong_region_prior_with_connected_component_mean_fill",
        postprocess_mode="connected_component_region_mean_fill",
        use_segmentation=True,
    ),
    "a3": VariantConfig(
        variant_id="a3",
        variant_name="A3 Main + Fixed Example Pair",
        description="Fixed RGB/metallic exemplar pair + target RGB, sharing the A0 metallic core.",
        input_mode="rgb_plus_example",
        route="seedream_metallic_a3_fixed_example",
        prompt_version="metallic_a3_fixed_example_v3_shared_core",
        example_pair_policy="fixed_global_exemplar_pair",
        use_example_pair=True,
    ),
}


def get_variant_config(variant_id: str) -> VariantConfig:
    key = str(variant_id or "").strip().lower()
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"Unsupported variant_id: {variant_id}. Expected one of: {sorted(VARIANT_CONFIGS)}")
    return VARIANT_CONFIGS[key]


def filter_images_by_names(image_paths: Sequence[Path], include_names: Optional[Sequence[str]]) -> List[Path]:
    if not include_names:
        return list(image_paths)

    wanted = [str(name).strip() for name in include_names if str(name).strip()]
    if not wanted:
        return list(image_paths)

    image_map: Dict[str, Path] = {}
    for path in image_paths:
        image_map[path.name] = path
        image_map[path.as_posix()] = path

    missing = [name for name in wanted if name not in image_map]
    if missing:
        raise FileNotFoundError(f"Requested image(s) not found in input_dir: {missing}")
    return [image_map[name] for name in wanted]


def infer_seg_dir_from_rgb_dir(rgb_dir: Path) -> Optional[Path]:
    parts = list(rgb_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "image":
            parts[idx] = "ObjectSegmentation"
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
    return [preferred] + [suffix for suffix in SEG_CANDIDATE_SUFFIXES if suffix != preferred]


def find_matching_seg(rgb_path: Path, seg_dir: Path, preferred_suffix: str = "") -> Path:
    stem = rgb_path.stem
    candidates: List[Path] = []
    seg_suffixes = get_seg_candidate_suffixes(preferred_suffix)
    candidate_dirs: List[Path] = [seg_dir]

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
        seg_name = "ObjectSegmentation_" + rgb_path.name[len("Image_") :]
        for candidate_dir in candidate_dirs:
            candidates.append(candidate_dir / seg_name)

    for tail in ["_im", "_rgb", "_image"]:
        if stem.lower().endswith(tail):
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


def file_sha256(path: Optional[Path]) -> str:
    if path is None:
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


A0_INPUT_HEADER = (
    "You are a senior PBR material analysis expert. "
    "You are given exactly one input image: the target RGB image. "
    "Generate only the target metallic map for that RGB image. "
)


def build_a0_prompt(prompt_preset: str = PROMPT_PRESET) -> str:
    return build_metallic_prompt(
        input_mode="rgb_only",
        prompt_preset=prompt_preset,
        scene_prompt="",
    )


def split_a0_header_and_core(prompt_preset: str = PROMPT_PRESET) -> Tuple[str, str]:
    a0_prompt = build_a0_prompt(prompt_preset=prompt_preset)
    if not a0_prompt.startswith(A0_INPUT_HEADER):
        raise RuntimeError(
            "A0 main prompt header changed in metallic_generation_doubao_final.py. "
            "Please update A0_INPUT_HEADER before running ablations."
        )
    return A0_INPUT_HEADER, a0_prompt[len(A0_INPUT_HEADER) :]


def build_prompt_from_shared_core(*, input_header: str, prior_block: str, prompt_preset: str = PROMPT_PRESET) -> str:
    _, shared_core = split_a0_header_and_core(prompt_preset=prompt_preset)
    return input_header + prior_block + shared_core


def soft_segmentation_prior_block() -> str:
    return (
        "The segmentation map is an oracle object/instance spatial prior used only for this ablation. "
        "It is not part of the RGB-only benchmark setting. "
        "Its colors are arbitrary region identifiers and do not encode semantic class, material type, metallic value, or confidence. "
        "Use it only to constrain the spatial support of filled material regions, reduce bleeding across object boundaries, and stabilize large-scale layout. "
        "Use segmentation to constrain filled regions, not to draw outlines. "
        "Do not draw segmentation boundaries, contours, seams, object outlines, or region edges as metallic responses. "
        "A single segmented object may contain both metal and non-metal parts, so do not force one metallic value per region. "
        "Preserve metal-versus-non-metal changes supported by the RGB image even when they fall inside the same segmentation region. "
        "When segmentation conflicts with visible RGB material evidence, trust the RGB image. "
    )


def strong_segmentation_prior_block() -> str:
    return (
        "The segmentation map is an oracle object/instance spatial prior used only for this ablation. "
        "It is not part of the RGB-only benchmark setting. "
        "The segmentation colors are arbitrary region identifiers and do not encode semantic class, material type, metallic value, or confidence. "
        "Use segmentation as a strong region-level spatial prior for large homogeneous material regions. "
        "Encourage connected-region coherence and stable region-level metallic assignment when the RGB image does not show a clear material change. "
        "However, do not erase small visible metallic subparts such as handles, hinges, knobs, brackets, screws, exposed metal trims, or other clearly exposed metal hardware when RGB evidence is clear. "
        "Prefer filled metallic regions over boundary-only responses, but preserve small filled metallic parts even if they lie inside a larger non-metal object region. "
        "Do not draw segmentation boundaries, object contours, seams, borders, or region edges as metallic lines. "
        "Use segmentation for spatial support and boundary placement only; do not treat segmentation color as a material cue. "
        "If visible RGB evidence strongly contradicts the segmentation prior, trust the RGB image. "
    )


def fixed_example_prior_block() -> str:
    return (
        "The reference RGB image and reference metallic map form one fixed exemplar pair shared across all target images in this experiment. "
        "Use the target RGB image as the primary evidence. "
        "Use the reference pair only as output-format, sparsity, near-binary metallic-map appearance, and spatial-alignment style guidance. "
        "The role of the reference RGB image is only to provide visual context for interpreting the reference metallic map; it is not a template for the target scene. "
        "Do not copy the reference scene layout, object placement, object boundaries, or semantic material identity onto the target scene. "
        "Do not transfer the reference object's metallic labels to visually different target objects. "
        "When the target RGB evidence conflicts with the reference pair, always trust the target RGB evidence. "
    )


def build_a1_prompt(prompt_preset: str = PROMPT_PRESET) -> str:
    return build_prompt_from_shared_core(
        prompt_preset=prompt_preset,
        input_header=(
            "You are a senior PBR material analysis expert. "
            "You are given exactly two input images in order: "
            "(1) target RGB image, "
            "(2) target object/instance segmentation prior map. "
            "Generate only the target metallic map for image (1). "
        ),
        prior_block=soft_segmentation_prior_block(),
    )


def build_a2_prompt(prompt_preset: str = PROMPT_PRESET) -> str:
    return build_prompt_from_shared_core(
        prompt_preset=prompt_preset,
        input_header=(
            "You are a senior PBR material analysis expert. "
            "You are given exactly two input images in order: "
            "(1) target RGB image, "
            "(2) target object/instance segmentation prior map. "
            "Generate only the target metallic map for image (1). "
        ),
        prior_block=strong_segmentation_prior_block(),
    )


def build_a3_prompt(prompt_preset: str = PROMPT_PRESET) -> str:
    return build_prompt_from_shared_core(
        prompt_preset=prompt_preset,
        input_header=(
            "You are a senior PBR material analysis expert. "
            "You are given exactly three input images in order: "
            "(1) reference RGB image, "
            "(2) reference metallic map, "
            "(3) target RGB image. "
            "Generate only the target metallic map for image (3). "
        ),
        prior_block=fixed_example_prior_block(),
    )


def build_prompt(variant_id: str, prompt_preset: str = PROMPT_PRESET) -> str:
    config = get_variant_config(variant_id)
    if config.variant_id == "a0":
        return build_a0_prompt(prompt_preset=prompt_preset)
    if config.variant_id == "a1":
        return build_a1_prompt(prompt_preset=prompt_preset)
    if config.variant_id == "a2":
        return build_a2_prompt(prompt_preset=prompt_preset)
    if config.variant_id == "a3":
        return build_a3_prompt(prompt_preset=prompt_preset)
    raise ValueError(f"Unsupported variant_id: {variant_id}")


def resolve_input_images(
    *,
    variant_id: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    example_rgb: Optional[Path],
    example_metallic: Optional[Path],
) -> List[Path]:
    config = get_variant_config(variant_id)
    if config.use_segmentation:
        if seg_path is None:
            raise ValueError(f"{variant_id} requires a matching segmentation image")
        return [rgb_path, seg_path]
    if config.use_example_pair:
        if example_rgb is None or example_metallic is None:
            raise ValueError(f"{variant_id} requires both example_rgb and example_metallic")
        return [example_rgb, example_metallic, rgb_path]
    return [rgb_path]


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

            if component_pixels:
                mean_value = float(np.mean([pred_np[py, px] for py, px in component_pixels]))
                for py, px in component_pixels:
                    output[py, px] = mean_value

    Image.fromarray(np.clip(np.round(output), 0, 255).astype(np.uint8), mode="L").save(save_path)


def build_run_signature(
    *,
    variant_id: str,
    image_model: str,
    prompt_text: str,
    prompt_preset: str,
    seg_dir: str,
    seg_suffix: str,
    example_rgb: str,
    example_metallic: str,
    example_rgb_sha256: str,
    example_metallic_sha256: str,
    timeout: int,
    size: str,
    watermark: bool,
) -> Dict[str, Any]:
    config = get_variant_config(variant_id)
    segmentation_assumption = "unused"
    if config.variant_id == "a1":
        segmentation_assumption = "oracle_object_instance_segmentation_as_soft_spatial_prior_only"
    elif config.variant_id == "a2":
        segmentation_assumption = "oracle_object_instance_segmentation_as_strong_region_prior_then_mean_fill_diagnostic"

    return {
        "variant_id": config.variant_id,
        "variant_name": config.variant_name,
        "image_model": image_model,
        "route": config.route,
        "input_mode": config.input_mode,
        "prompt_version": config.prompt_version,
        "prompt_preset": prompt_preset,
        "prompt_text": prompt_text,
        "segmentation_role": config.segmentation_role,
        "segmentation_assumption": segmentation_assumption,
        "seg_dir": seg_dir,
        "seg_suffix": normalize_seg_suffix(seg_suffix),
        "postprocess_mode": config.postprocess_mode,
        "example_pair_policy": config.example_pair_policy,
        "example_rgb": example_rgb,
        "example_metallic": example_metallic,
        "example_rgb_sha256": example_rgb_sha256,
        "example_metallic_sha256": example_metallic_sha256,
        "timeout": timeout,
        "size": size,
        "output_resolution_policy": "model_native_output",
        "watermark": watermark,
    }


def should_skip_existing_output(
    *,
    metallic_path: Path,
    output_meta_path: Path,
    run_signature: Dict[str, Any],
    input_fingerprints: Dict[str, Any],
) -> bool:
    if not metallic_path.exists() or not output_meta_path.exists():
        return False
    saved = load_json_dict(output_meta_path)
    return saved.get("run_signature") == run_signature and saved.get("input_fingerprints") == input_fingerprints


def generate_one_case(
    *,
    ark_client: Any,
    model: str,
    variant_id: str,
    rgb_path: Path,
    seg_path: Optional[Path],
    save_path: Path,
    example_rgb: Optional[Path],
    example_metallic: Optional[Path],
    size: str,
    watermark: bool,
    timeout: int,
    prompt_preset: str = PROMPT_PRESET,
    source_prediction_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = get_variant_config(variant_id)
    prompt = build_prompt(variant_id, prompt_preset=prompt_preset)
    image_paths = resolve_input_images(
        variant_id=variant_id,
        rgb_path=rgb_path,
        seg_path=seg_path,
        example_rgb=example_rgb,
        example_metallic=example_metallic,
    )

    source_prediction = None
    if config.postprocess_mode == "connected_component_region_mean_fill":
        if seg_path is None:
            raise ValueError(f"{variant_id} requires a matching segmentation image")
        source_prediction = source_prediction_path or save_path.with_name(f"{save_path.stem}_source.png")
        source_prediction.parent.mkdir(parents=True, exist_ok=True)
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
        "example_metallic": str(example_metallic) if example_metallic else "",
    }
    if source_prediction is not None:
        result["source_prediction"] = source_prediction.as_posix()
    return result


def main() -> None:
    raise RuntimeError("Use metallic_ablation_runner_strict_final.py to run A0/A1/A2/A3 ablations.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
