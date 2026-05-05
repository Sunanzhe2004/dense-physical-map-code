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
DEFAULT_NORMAL_MODEL = "doubao-seedream-5-0-260128"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    variant_name: str
    prompt_version: str
    description: str
    prompt_text: str
    use_example_pair: bool
    prompt_level: str
    default_example_rgb: str = ""
    default_example_normal: str = ""


def parse_args(config: VariantConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--filename_suffix", type=str, default="_im.png")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--normal_model", type=str, default=DEFAULT_NORMAL_MODEL)
    parser.add_argument("--normal_size", type=str, default="adaptive")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--max_generate", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preserve_relative_dirs", action="store_true")
    parser.add_argument("--max_retries", type=int, default=3, help="Retry count per image on generation failure.")
    parser.add_argument("--retry_sleep", type=float, default=3.0, help="Sleep seconds between retries.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--example_rgb",
        type=str,
        default="",
        help="Optional override for the exemplar RGB used by fixed-example variants.",
    )
    parser.add_argument(
        "--example_normal",
        type=str,
        default="",
        help="Optional override for the exemplar normal map used by fixed-example variants.",
    )
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY in environment.")
    return api_key


def list_images(input_dir: Path, filename_suffix: Optional[str], recursive: bool) -> List[Path]:
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


def save_url_to_file(url: str, save_path: Path, timeout: int) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int) -> None:
    url = getattr(image_item, "url", None)
    if url:
        save_url_to_file(url, save_path, timeout)
        return

    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        save_path.write_bytes(base64.b64decode(b64_json))
        return

    if isinstance(image_item, dict):
        if image_item.get("url"):
            save_url_to_file(image_item["url"], save_path, timeout)
            return
        if image_item.get("b64_json"):
            save_path.write_bytes(base64.b64decode(image_item["b64_json"]))
            return

    raise RuntimeError("Image response contains neither url nor b64_json.")


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


def upsert_manifest_entry(
    manifest: List[Dict[str, Any]],
    entry: Dict[str, Any],
    key_field: str = "relative_image_path",
) -> None:
    entry_key = entry.get(key_field)
    if entry_key is None:
        manifest.append(entry)
        return
    for idx, item in enumerate(manifest):
        if item.get(key_field) == entry_key:
            manifest[idx] = entry
            return
    manifest.append(entry)


def build_skip_path(meta_dir: Path) -> Path:
    return meta_dir / "skipped_images.json"


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    config: VariantConfig,
    exc: Exception,
    attempts: int,
) -> None:
    entry = {
        "skip_key": f"{config.variant_id}::{image_path.relative_to(input_dir).as_posix()}",
        "variant_id": config.variant_id,
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "attempts": attempts,
        "skipped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    upsert_manifest_entry(skipped_images, entry, key_field="skip_key")
    skipped_path.write_text(json.dumps(skipped_images, ensure_ascii=False, indent=2), encoding="utf-8")


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
    normal_dir: Path,
    overwrite: bool,
    preserve_relative_dirs: bool,
) -> List[Path]:
    if overwrite:
        return list(image_paths)
    pending: List[Path] = []
    for image_path in image_paths:
        out_normal = build_image_output_path(
            image_path,
            input_dir,
            normal_dir,
            "_normal.png",
            preserve_relative_dirs,
        )
        if is_completed_output(out_normal):
            continue
        pending.append(image_path)
    return pending


def build_setup_payload(config: VariantConfig, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "variant_id": config.variant_id,
        "variant_name": config.variant_name,
        "prompt_version": config.prompt_version,
        "prompt_level": config.prompt_level,
        "use_example_pair": config.use_example_pair,
        "prompt_text": config.prompt_text,
        "normal_model": args.normal_model,
        "base_url": args.base_url,
        "normal_size": args.normal_size,
        "watermark": args.watermark,
        "max_generate": args.max_generate,
        "overwrite": args.overwrite,
        "preserve_relative_dirs": args.preserve_relative_dirs,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "filename_suffix": args.filename_suffix,
        "recursive": args.recursive,
        "example_rgb": args.example_rgb if config.use_example_pair else "",
        "example_normal": args.example_normal if config.use_example_pair else "",
    }


def run_image_generation(
    ark_client: Any,
    model: str,
    prompt: str,
    image_paths: List[Path],
    size: str,
    watermark: bool,
) -> Any:
    image_uris = [file_to_data_uri(path) for path in image_paths]
    normalized_size = (size or "adaptive").strip().lower()
    if normalized_size == "adaptive":
        normalized_size = "2k"

    if "seedream" in model.lower():
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris,
            size=normalized_size,
            watermark=watermark,
        )

    try:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris if len(image_uris) > 1 else image_uris[0],
            size=normalized_size,
            watermark=watermark,
        )
    except TypeError:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris[-1],
            size=normalized_size,
            watermark=watermark,
        )


def run_variant(config: VariantConfig) -> None:
    args = parse_args(config)
    api_key = ensure_api_key()

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install: pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    normal_dir = output_dir / "normal"
    meta_dir = output_dir / "meta"
    normal_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    examples_dir = script_dir / "examples"
    example_rgb = None
    example_normal = None
    if config.use_example_pair:
        example_rgb = Path(args.example_rgb).expanduser() if args.example_rgb else (
            examples_dir / config.default_example_rgb if config.default_example_rgb else None
        )
        example_normal = Path(args.example_normal).expanduser() if args.example_normal else (
            examples_dir / config.default_example_normal if config.default_example_normal else None
        )
    if config.use_example_pair:
        if not example_rgb or not example_normal:
            raise ValueError("This variant requires both --example_rgb and --example_normal")
        if not example_rgb.exists():
            raise FileNotFoundError(f"example_rgb not found: {example_rgb}")
        if not example_normal.exists():
            raise FileNotFoundError(f"example_normal not found: {example_normal}")

    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    pending_image_paths = get_pending_images(
        image_paths_for_generate,
        input_dir,
        normal_dir,
        args.overwrite,
        args.preserve_relative_dirs,
    )

    manifest_path = meta_dir / "manifest.json"
    skipped_path = build_skip_path(meta_dir)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)
    setup_path = meta_dir / "setup.json"
    setup_path.write_text(
        json.dumps(build_setup_payload(config, args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[1/3] found {len(image_paths)} images; selected {len(image_paths_for_generate)}; "
        f"pending {len(pending_image_paths)}; variant={config.variant_id}; "
        f"prompt_level={config.prompt_level}; example_pair={config.use_example_pair}"
    )
    print("[2/3] fixed prompt ablation mode; analysis disabled")
    print(f"[3/3] start normal generation for {len(pending_image_paths)} pending images")

    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    for idx, image_path in enumerate(image_paths_for_generate, start=1):
        out_normal = build_image_output_path(
            image_path,
            input_dir,
            normal_dir,
            "_normal.png",
            args.preserve_relative_dirs,
        )
        if not args.overwrite and is_completed_output(out_normal):
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} exists, skip")
            continue

        print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")
        input_images = [image_path]
        if config.use_example_pair and example_rgb and example_normal:
            input_images = [example_rgb, example_normal, image_path]

        last_exc: Optional[Exception] = None
        success = False
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                response = run_image_generation(
                    ark_client=ark_client,
                    model=args.normal_model,
                    prompt=config.prompt_text,
                    image_paths=input_images,
                    size=args.normal_size,
                    watermark=args.watermark,
                )
                ensure_parent_dir(out_normal)
                save_image_response(response.data[0], out_normal, timeout=args.timeout)
                success = True
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= max(1, args.max_retries):
                    break
                print(
                    f"      generation failed on attempt {attempt}/{max(1, args.max_retries)}: {exc} | retrying"
                )
                time.sleep(max(0.0, args.retry_sleep))

        if not success:
            assert last_exc is not None
            print(
                f"      skip after {max(1, args.max_retries)} failed attempt(s): {last_exc}"
            )
            record_skipped_image(
                skipped_images=skipped_images,
                skipped_path=skipped_path,
                image_path=image_path,
                input_dir=input_dir,
                config=config,
                exc=last_exc,
                attempts=max(1, args.max_retries),
            )
            continue

        per_image_meta = {
            "image_name": image_path.name,
            "relative_image_path": image_path.relative_to(input_dir).as_posix(),
            "variant_id": config.variant_id,
            "variant_name": config.variant_name,
            "prompt_version": config.prompt_version,
            "prompt_level": config.prompt_level,
            "use_example_pair": config.use_example_pair,
            "normal_model": args.normal_model,
            "normal_output": out_normal.relative_to(output_dir).as_posix(),
            "normal_prompt": config.prompt_text,
            "example_rgb": str(example_rgb) if example_rgb else "",
            "example_normal": str(example_normal) if example_normal else "",
        }
        upsert_manifest_entry(manifest, per_image_meta)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(max(0.0, args.sleep))

    print("Done.")
    print(f"Normal output dir: {normal_dir.as_posix()}")
    print(f"Meta output dir: {meta_dir.as_posix()}")


FULL_PROMPT_CORE = (
    "Generate a strict view-space per-pixel indoor normal map. "
    "Preserve the input scene layout, object boundaries, silhouettes, broad curvature, panel recesses, frame thickness, and real shape transitions. "
    "Use RGB only to encode surface orientation. "
    "Do not preserve color, lighting tint, shading, highlights, reflections, cast shadows, ambient occlusion, or material texture. "
    "Suppress wood grain, fabric weave, printed patterns, gloss streaks, and image noise. "
    "Large planar regions should stay smooth and stable. "
    "Curtains, blankets, pillows, and sofas should keep only broad macro folds. "
    "Do not invent bevels, grooves, contours, or extra small detail. "
)


FULL_RGB_PLUS_EXAMPLE_PROMPT = (
    "Input: (1) example RGB, (2) example normal map, (3) target RGB. "
    "Task: generate the normal map for the target RGB image. "
    "Use the example normal map only as the encoding and color-family reference. "
    + FULL_PROMPT_CORE
    + "Output: one normal map image."
)


FULL_RGB_ONLY_PROMPT = (
    "Input: target RGB. "
    "Task: generate the normal map for the input RGB image. "
    + FULL_PROMPT_CORE
    + "Output: one normal map image."
)


MINIMAL_RGB_PLUS_EXAMPLE_PROMPT = (
    "Input: (1) example RGB, (2) example normal map, (3) target RGB. "
    "Task: generate the normal map for the target RGB image. "
    "Output: one normal map image."
)


MINIMAL_RGB_ONLY_PROMPT = (
    "Task: generate the normal map of the input RGB image. "
    "Output: one normal map image."
)


def main() -> None:
    raise RuntimeError("Import this module from a specific ablation script instead of running it directly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
