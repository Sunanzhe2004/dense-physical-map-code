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

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ALBEDO_MODEL = "doubao-seedream-5-0-260128"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    prompt_version: str
    description: str
    prompt_text: str
    use_example_pair: bool = False
    prompt_level: str = "fixed"


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_suffix", type=str, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input_dir.")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--albedo_model", type=str, default=DEFAULT_ALBEDO_MODEL)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--albedo_size", type=str, default="adaptive")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--part_index", type=int, default=0)
    parser.add_argument(
        "--preserve_relative_dirs",
        action="store_true",
        help="Preserve the relative directory structure under input_dir.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--example_rgb",
        type=str,
        required=config.use_example_pair,
        default="",
        help="Required for variants that use a fixed exemplar RGB/albedo pair.",
    )
    parser.add_argument(
        "--example_albedo",
        type=str,
        required=config.use_example_pair,
        default="",
        help="Required for variants that use a fixed exemplar RGB/albedo pair.",
    )
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
        raise FileNotFoundError(f"No supported images found in {input_dir}")
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


def save_url_to_file(url: str, save_path: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
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

    if isinstance(image_item, dict):
        if image_item.get("url"):
            save_url_to_file(image_item["url"], save_path, timeout=timeout)
            return
        if image_item.get("b64_json"):
            save_path.write_bytes(base64.b64decode(image_item["b64_json"]))
            return

    raise RuntimeError("Image response contains neither url nor b64_json.")


def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
) -> Any:
    image_data_uris = [file_to_data_uri(path) for path in image_paths]
    model_lower = model.lower()
    normalized_size = (size or "").strip().lower()
    if not normalized_size or normalized_size == "adaptive":
        normalized_size = "2k"

    if "seedream" in model_lower:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_data_uris,
            size=normalized_size,
            watermark=watermark,
        )

    try:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_data_uris if len(image_data_uris) > 1 else image_data_uris[0],
            seed=seed,
            guidance_scale=guidance_scale,
            size=normalized_size,
            watermark=watermark,
        )
    except TypeError:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_data_uris[-1],
            size=normalized_size,
            watermark=watermark,
        )


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


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


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


def build_setup_payload(config: VariantConfig, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "variant_id": config.variant_id,
        "variant_name": config.variant_name,
        "prompt_version": config.prompt_version,
        "prompt_level": config.prompt_level,
        "analysis_enabled": False,
        "use_example_pair": config.use_example_pair,
        "prompt_text": config.prompt_text,
        "albedo_model": args.albedo_model,
        "base_url": args.base_url,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "albedo_size": args.albedo_size,
        "watermark": args.watermark,
        "max_generate": args.max_generate,
        "overwrite": args.overwrite,
        "num_parts": args.num_parts,
        "part_index": args.part_index,
        "preserve_relative_dirs": args.preserve_relative_dirs,
        "filename_suffix": args.filename_suffix,
        "recursive": args.recursive,
        "example_rgb": args.example_rgb if config.use_example_pair else "",
        "example_albedo": args.example_albedo if config.use_example_pair else "",
    }


def run_variant(config: VariantConfig) -> None:
    args = parse_args(config)
    validate_parts(args.num_parts, args.part_index)
    api_key = ensure_api_key(args.api_key)

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    albedo_dir = output_dir / "albedo"
    meta_dir = output_dir / "meta"
    per_image_meta_dir = meta_dir / "per_image"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    per_image_meta_dir.mkdir(parents=True, exist_ok=True)

    example_rgb = Path(args.example_rgb) if config.use_example_pair else None
    example_albedo = Path(args.example_albedo) if config.use_example_pair else None
    if config.use_example_pair:
        if not example_rgb or not example_albedo:
            raise ValueError("This variant requires both --example_rgb and --example_albedo")
        if not example_rgb.exists():
            raise FileNotFoundError(f"example_rgb not found: {example_rgb}")
        if not example_albedo.exists():
            raise FileNotFoundError(f"example_albedo not found: {example_albedo}")

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

    print(
        f"[1/3] found {len(image_paths)} images; current {shard_label} handles "
        f"{len(image_paths_for_generate)} images; pending {len(pending_image_paths)}; "
        f"variant={config.variant_id} ({config.variant_name}) | "
        f"prompt_level={config.prompt_level} | example_pair={config.use_example_pair}"
    )
    print(
        "      "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"albedo_model={args.albedo_model} | "
        f"base_url={args.base_url}"
    )

    setup_path = meta_dir / "setup.json"
    setup_path.write_text(
        json.dumps(build_setup_payload(config, args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_path = meta_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    ark_client = Ark(base_url=args.base_url, api_key=api_key)

    if not pending_image_paths:
        print("[2/3] nothing to do; all outputs already exist for this shard")
        print("[3/3] done")
        print(f"albedo output dir: {albedo_dir.as_posix()}")
        print(f"meta output dir: {meta_dir.as_posix()}")
        return

    print("[2/3] analysis disabled for this ablation; using fixed prompt only")
    print(f"[3/3] start albedo generation for {len(pending_image_paths)} images")

    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_albedo = build_image_output_path(
            image_path,
            input_dir,
            albedo_dir,
            "_albedo.png",
            args.preserve_relative_dirs,
        )
        out_meta = build_image_output_path(
            image_path,
            input_dir,
            per_image_meta_dir,
            "_albedo.json",
            args.preserve_relative_dirs,
        )

        if not args.overwrite and is_completed_output(out_albedo):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} exists, skip")
            continue

        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")
        input_images = [image_path]
        if config.use_example_pair and example_rgb and example_albedo:
            input_images = [example_rgb, example_albedo, image_path]
        albedo_resp = run_image_generation(
            ark_client,
            args.albedo_model,
            config.prompt_text,
            input_images,
            args.albedo_size,
            args.watermark,
            args.seed,
            args.guidance_scale,
        )
        ensure_parent_dir(out_albedo)
        save_image_response(albedo_resp.data[0], out_albedo, timeout=args.timeout)

        per_image_meta = {
            "image_name": image_path.name,
            "relative_image_path": image_path.relative_to(input_dir).as_posix(),
            "variant_id": config.variant_id,
            "variant_name": config.variant_name,
            "prompt_version": config.prompt_version,
            "prompt_level": config.prompt_level,
            "analysis_enabled": False,
            "use_example_pair": config.use_example_pair,
            "albedo_model": args.albedo_model,
            "albedo_output": out_albedo.relative_to(output_dir).as_posix(),
            "albedo_prompt": config.prompt_text,
            "example_rgb": str(example_rgb) if example_rgb else "",
            "example_albedo": str(example_albedo) if example_albedo else "",
            "num_parts": args.num_parts,
            "part_index": args.part_index,
        }
        ensure_parent_dir(out_meta)
        out_meta.write_text(json.dumps(per_image_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        upsert_manifest_entry(
            manifest,
            {
                **per_image_meta,
                "per_image_meta": out_meta.relative_to(output_dir).as_posix(),
            },
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(max(0.0, args.sleep))

    print("done")
    print(f"albedo output dir: {albedo_dir.as_posix()}")
    print(f"meta output dir: {meta_dir.as_posix()}")


FULL_PROMPT_TEXT = (
    "Convert the input image into a clean intrinsic albedo image. "
    "Preserve the exact scene layout, visible objects, object boundaries, and material regions. "
    "Remove illumination effects only: cast shadows, attached shading, highlights, reflections, "
    "interreflections, exposure variation, ambient occlusion residue, and illumination color cast. "
    "Keep the intrinsic reflectance or base color of each visible surface. "
    "Preserve low-frequency and mid-frequency material texture that belongs to the surface itself, "
    "such as wood color variation, subtle fabric color variation, printed material color, and "
    "material-region color boundaries, but do not preserve lighting-induced brightness gradients. "
    "Do not simplify the image into flat poster-like color blocks unless the original surface is truly uniform. "
    "Do not add new objects. Do not remove existing objects. Do not stylize. Do not change geometry. "
    "Do not perform completion. Output a clean, illumination-free, texture-preserving albedo map."
)


WEAKENED_PROMPT_TEXT = (
    "Convert the input image into an intrinsic albedo or reflectance-like image. "
    "Preserve the exact scene layout, visible objects, object boundaries, and intrinsic material texture. "
    "Reduce illumination-dependent appearance as much as possible while keeping the surface's own color patterns. "
    "Keep material-region boundaries and real surface color variation. "
    "Do not add new objects. Do not remove existing objects. Do not stylize. Do not change geometry. "
    "Output a structure-preserving albedo image."
)


MINIMAL_PROMPT_TEXT = (
    "Generate the intrinsic albedo or reflectance map of the input image. "
    "Preserve the scene structure, visible objects, and object boundaries. "
    "Do not add or remove objects. Output one albedo image only."
)


MINIMAL_FIXED_EXEMPLAR_PROMPT_TEXT = (
    "Input: (1) example RGB, (2) example intrinsic albedo, (3) target RGB. "
    "Task: generate the intrinsic albedo map for the target RGB image. "
    "Use the example RGB/albedo pair as a reference sample for translating appearance into intrinsic albedo. "
    "Preserve the target scene structure, visible objects, and object boundaries. "
    "Do not add or remove objects. Output one albedo image only."
)


def main() -> None:
    raise RuntimeError("Import this module from a specific ablation script instead of running it directly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
