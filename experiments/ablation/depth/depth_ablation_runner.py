#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
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
DEPTH_POLARITY = "near_white"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    description: str
    prompt_version: str
    input_mode: str
    route: str
    evaluation_protocol: str
    evaluation_note: str
    use_example_pair: bool = False
    use_segmentation: bool = False


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input_dir", type=str, required=True, help="RGB image directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Ark base URL.")
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model.")
    parser.add_argument("--size", type=str, default="adaptive", help="Output size; adaptive maps to 2k.")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests.")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images with matching outputs and signatures.")
    parser.add_argument(
        "--example_rgb",
        type=str,
        default="",
        help="Optional override for the fixed exemplar RGB used by rgb_plus_example variants.",
    )
    parser.add_argument(
        "--example_depth",
        type=str,
        default="",
        help="Optional override for the fixed exemplar depth map used by rgb_plus_example variants.",
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


def list_images(input_dir: Path, *, seg_mode: bool) -> List[Path]:
    if seg_mode:
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"input_dir does not exist or is not a directory: {input_dir}")
        images = [
            p for p in sorted(input_dir.iterdir())
            if p.is_file() and p.suffix.lower() == ".png" and p.stem.lower().endswith("_im")
        ]
        if not images:
            raise FileNotFoundError(f"No RGB images matching *_im.png were found in {input_dir}")
        return images

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


def get_paired_seg_path(rgb_path: Path) -> Path:
    if not rgb_path.stem.lower().endswith("_im"):
        raise ValueError(f"RGB image name must end with _im before extension: {rgb_path.name}")
    seg_stem = rgb_path.stem[:-3] + "_seg"
    return rgb_path.with_name(seg_stem + rgb_path.suffix)


def build_common_depth_body() -> str:
    return (
        "This is a relative depth prediction task, not metric depth regression. "
        "Predict camera-centric scene depth up to an unknown affine transform. "
        "The output must be a single-channel grayscale depth-buffer-like map, pixel-aligned with the target RGB image. "
        "Do not crop, zoom, shift, recompose, add, remove, or move any scene content. "
        "White means closer to the camera. Black means farther from the camera. "
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


def build_relative_depth_prompt(config: VariantConfig) -> str:
    common_body = build_common_depth_body()
    if config.input_mode == "rgb_only":
        return (
            "You are a senior monocular depth estimation and 3D scene geometry expert. "
            "You are given exactly one input image: the target RGB image. "
            "Generate only a dense single-image relative depth map for that RGB image. "
            + common_body
        )
    if config.input_mode == "rgb_plus_example":
        return (
            "You are a senior monocular depth estimation and 3D scene geometry expert. "
            "You are given exactly three input images in order: "
            "(1) reference RGB image, "
            "(2) reference relative depth map, "
            "(3) target RGB image. "
            "Generate only the dense relative depth map for image (3). "
            "Use image (2) only to infer the polarity convention, clean grayscale label style, and desired degree of piecewise smoothness. "
            "Do not copy or transfer the reference scene layout, object placement, object geometry, room geometry, global brightness histogram, contrast distribution, depth range compression, local shading pattern, edge softness, or scene-specific tonal balance onto the target image. "
            "The target depth distribution must be determined only from the target RGB image's geometry. "
            "Do not normalize the target output to imitate the reference depth map's overall brightness distribution or contrast profile. "
            "If the reference depth map and the target RGB image suggest different depth distributions, follow the target RGB image only. "
            + common_body
        )
    if config.input_mode == "rgb_plus_seg":
        return (
            "You are a senior monocular depth estimation and 3D scene geometry expert. "
            "You are given exactly two input images in order: "
            "(1) target RGB image, "
            "(2) target segmentation map aligned pixel-to-pixel with the target RGB image. "
            "Generate only the dense relative depth map for image (1). "
            "Use the segmentation map only as a weak spatial prior for object extents, surface regions, silhouettes, occlusion boundaries, and small or thin structure preservation. "
            "The segmentation map is not a depth map. The colors or IDs in the segmentation map are arbitrary labels and must never be interpreted as brightness, material, texture, illumination, metric depth, or ordinal depth. "
            "Do not copy segmentation colors into the output. "
            "Do not assign a constant depth to an entire segment unless the visible geometry is actually fronto-parallel and depth-constant. "
            "For one segment covering a slanted or extended object, depth should still vary according to the object's 3D shape and orientation. "
            "If the segmentation boundary and RGB evidence conflict, follow the visible RGB geometry while using segmentation only to reduce boundary bleeding. "
            + common_body
        )
    raise ValueError(f"Unsupported input_mode: {config.input_mode}")


def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
) -> Any:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    image_uris = [file_to_data_uri(p) for p in image_paths]
    normalized_size = (size or "").strip().lower() or "adaptive"
    if normalized_size == "adaptive":
        normalized_size = "2k"
    return ark_client.images.generate(
        model=model,
        prompt=prompt,
        image=image_uris,
        size=normalized_size,
        watermark=watermark,
    )


def save_url_to_file(url: str, save_path: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


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


def postprocess_depth_png(path: Path) -> None:
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
    config: VariantConfig,
    image_model: str,
    prompt_text: str,
    timeout: int,
    size: str,
    watermark: bool,
    example_rgb: str = "",
    example_depth: str = "",
) -> Dict[str, Any]:
    signature = {
        "image_model": image_model,
        "route": config.route,
        "input_mode": config.input_mode,
        "prompt_version": config.prompt_version,
        "prompt_text": prompt_text,
        "depth_representation": "relative_depth",
        "depth_polarity": DEPTH_POLARITY,
        "evaluation_protocol": config.evaluation_protocol,
        "evaluation_note": config.evaluation_note,
        "timeout": timeout,
        "size": size,
        "input_resolution_policy": "source_images_without_padding",
        "output_resolution_policy": "model_native_output_without_crop_or_resize",
        "watermark": watermark,
    }
    if config.use_example_pair:
        signature["example_rgb"] = example_rgb
        signature["example_depth"] = example_depth
    if config.use_segmentation:
        signature["segmentation_policy"] = "paired_same_dir_suffix_im_to_seg"
    return signature


def should_skip_existing_output(*, depth_path: Path, output_meta_path: Path, run_signature: Dict[str, Any]) -> bool:
    if not depth_path.exists() or not output_meta_path.exists():
        return False
    saved = load_json_dict(output_meta_path)
    return saved.get("run_signature") == run_signature


def get_example_paths(args: argparse.Namespace, script_dir: Path) -> tuple[Path, Path]:
    example_dir = script_dir / "examples"
    example_rgb = Path(args.example_rgb).expanduser() if args.example_rgb else example_dir / "image.png"
    example_depth = Path(args.example_depth).expanduser() if args.example_depth else example_dir / "depth.png"
    for path_obj, name in ((example_rgb, "example_rgb"), (example_depth, "example_depth")):
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"{name} not found: {path_obj}")
    return example_rgb, example_depth


def generate_relative_depth_map(
    *,
    ark_client: Any,
    config: VariantConfig,
    model: str,
    rgb_path: Path,
    save_path: Path,
    size: str,
    watermark: bool,
    timeout: int,
    prompt: str,
    example_rgb: Optional[Path] = None,
    example_depth: Optional[Path] = None,
    seg_path: Optional[Path] = None,
) -> None:
    image_paths: List[Path] = [rgb_path]
    if config.use_example_pair:
        if example_rgb is None or example_depth is None:
            raise ValueError("Example-based depth variant requires example_rgb and example_depth.")
        image_paths = [example_rgb, example_depth, rgb_path]
    elif config.use_segmentation:
        if seg_path is None:
            raise ValueError("Segmentation-based depth variant requires seg_path.")
        image_paths = [rgb_path, seg_path]

    response = run_image_generation(
        ark_client=ark_client,
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        size=size,
        watermark=watermark,
    )
    save_image_response(response.data[0], save_path, timeout=timeout)
    postprocess_depth_png(save_path)


def run_variant(config: VariantConfig, script_dir: Path) -> None:
    args = parse_args(config)
    api_key = ensure_api_key()

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    input_dir = Path(args.input_dir).expanduser()

    if not config.use_segmentation and looks_like_object_seg_dir(input_dir):
        inferred_rgb_dir = infer_rgb_dir_from_seg_dir(input_dir)
        if inferred_rgb_dir is None:
            raise RuntimeError(
                f"input_dir looks like an ObjectSegmentation directory but the paired RGB directory could not be inferred: {input_dir}"
            )
        print(f"[info] detected segmentation directory as input_dir, switching RGB directory to: {inferred_rgb_dir}")
        input_dir = inferred_rgb_dir

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "relative_depth"
    meta_dir = output_dir / "meta"
    output_meta_dir = meta_dir / "per_image"
    depth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir, seg_mode=config.use_segmentation)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths

    example_rgb: Optional[Path] = None
    example_depth: Optional[Path] = None
    if config.use_example_pair:
        example_rgb, example_depth = get_example_paths(args, script_dir)

    prompt_text = build_relative_depth_prompt(config)
    run_signature = build_run_signature(
        config=config,
        image_model=args.image_model,
        prompt_text=prompt_text,
        timeout=args.timeout,
        size=args.size,
        watermark=args.watermark,
        example_rgb=str(example_rgb) if example_rgb else "",
        example_depth=str(example_depth) if example_depth else "",
    )
    (meta_dir / "setup.json").write_text(json.dumps(run_signature, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = meta_dir / "manifest.json"
    manifest: List[Dict[str, Any]] = load_manifest(manifest_path)

    print(
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} relative depth maps "
        f"with model={args.image_model}, input_mode={config.input_mode}, depth_polarity={DEPTH_POLARITY}"
    )

    print(f"[2/3] start generation for {len(image_paths_for_generate)} images")
    for idx, rgb_path in enumerate(image_paths_for_generate, start=1):
        depth_path = depth_dir / f"{rgb_path.stem}_relative_depth.png"
        output_meta_path = output_meta_dir / f"{rgb_path.stem}_relative_depth.json"
        item: Dict[str, Any] = {
            "image_name": rgb_path.name,
            "prompt_version": config.prompt_version,
            "input_mode": config.input_mode,
            "depth_representation": "relative_depth",
            "depth_polarity": DEPTH_POLARITY,
        }

        seg_path: Optional[Path] = None
        if config.use_segmentation:
            seg_path = get_paired_seg_path(rgb_path)
            item["segmentation_name"] = seg_path.name

        if config.use_example_pair and example_rgb and example_depth:
            item["example_rgb"] = example_rgb.name
            item["example_depth"] = example_depth.name

        try:
            if seg_path is not None and (not seg_path.exists() or not seg_path.is_file()):
                raise FileNotFoundError(f"Paired segmentation map not found: {seg_path}")

            progress_line = f"  - ({idx}/{len(image_paths_for_generate)}) {rgb_path.name}"
            if seg_path is not None:
                progress_line += f" | seg={seg_path.name}"
            elif example_rgb is not None:
                progress_line += f" | example={example_rgb.name}"
            print(progress_line)

            if args.skip_existing and should_skip_existing_output(
                depth_path=depth_path,
                output_meta_path=output_meta_path,
                run_signature=run_signature,
            ):
                item["skipped"] = True
                item["skip_reason"] = "matching_output_and_signature"
                item["relative_depth_output"] = depth_path.name
            else:
                generate_relative_depth_map(
                    ark_client=ark_client,
                    config=config,
                    model=args.image_model,
                    rgb_path=rgb_path,
                    save_path=depth_path,
                    size=args.size,
                    watermark=args.watermark,
                    timeout=args.timeout,
                    prompt=prompt_text,
                    example_rgb=example_rgb,
                    example_depth=example_depth,
                    seg_path=seg_path,
                )
                item["relative_depth_output"] = depth_path.name
                output_meta: Dict[str, Any] = {
                    "image_name": rgb_path.name,
                    "relative_depth_output": depth_path.name,
                    "input_mode": config.input_mode,
                    "run_signature": run_signature,
                }
                if seg_path is not None:
                    output_meta["segmentation_name"] = seg_path.name
                if example_rgb is not None and example_depth is not None:
                    output_meta["example_rgb"] = str(example_rgb)
                    output_meta["example_depth"] = str(example_depth)
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
