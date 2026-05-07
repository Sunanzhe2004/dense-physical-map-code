#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Doubao metallic ablations A0/A1/A2/A3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from metallic_generation_ablation_strict_final import (
    Ark,
    DEFAULT_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TIMEOUT,
    PROMPT_PRESET,
    _ARK_IMPORT_ERROR,
    build_prompt,
    build_run_signature,
    ensure_api_key,
    file_sha256,
    filter_images_by_names,
    find_matching_seg,
    generate_one_case,
    get_variant_config,
    infer_rgb_dir_from_seg_dir,
    infer_seg_dir_from_rgb_dir,
    list_images,
    load_manifest,
    looks_like_object_seg_dir,
    normalize_seg_suffix,
    should_skip_existing_output,
    upsert_manifest_entry,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Doubao metallic ablations for A0/A1/A2/A3.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--variant", type=str, default="", help="Single variant to run: a0/a1/a2/a3.")
    group.add_argument("--variants", nargs="+", default=None, help="One or more variants to run.")
    parser.add_argument("--input_dir", type=str, required=True, help="Target RGB image directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Ablation root output directory.")
    parser.add_argument("--seg_dir", type=str, default="", help="Segmentation directory for a1/a2. If omitted, try to infer ObjectSegmentation from input_dir.")
    parser.add_argument("--example_rgb", type=str, default="", help="Reference RGB for a3.")
    parser.add_argument("--example_metallic", type=str, default="", help="Reference metallic map for a3.")
    parser.add_argument("--include_names", nargs="*", default=None, help="Optional explicit filenames to process.")
    parser.add_argument("--repeat_index", type=int, default=0, help="Repeat index recorded in metadata.")
    parser.add_argument("--prompt_preset", type=str, default=PROMPT_PRESET, help="Prompt preset used by A0 main prompt builder.")
    parser.add_argument("--seg_suffix", type=str, default="", help="Preferred segmentation suffix.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Ark base URL.")
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image model.")
    parser.add_argument("--size", type=str, default="adaptive", help="Output size; adaptive -> 2k.")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request/download timeout.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests.")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip matching completed outputs.")
    return parser.parse_args()


def resolve_variant_ids(args: argparse.Namespace) -> List[str]:
    raw_variants = args.variants if args.variants is not None else ([args.variant] if args.variant else [])
    if not raw_variants:
        raw_variants = ["a0", "a1", "a2", "a3"]
    variant_ids = [str(variant).strip().lower() for variant in raw_variants if str(variant).strip()]
    if not variant_ids:
        raise ValueError("No valid variants requested.")
    for variant_id in variant_ids:
        get_variant_config(variant_id)
    return variant_ids


def ensure_example_pair(example_rgb: Optional[Path], example_metallic: Optional[Path]) -> None:
    for path_obj, name in ((example_rgb, "example_rgb"), (example_metallic, "example_metallic")):
        if path_obj is None or not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"{name} not found: {path_obj}")


def resolve_default_example_pair(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path]]:
    script_dir = Path(__file__).resolve().parent
    examples_dir = script_dir / "examples"
    example_rgb = Path(args.example_rgb).expanduser() if args.example_rgb else examples_dir / "image.png"
    example_metallic = Path(args.example_metallic).expanduser() if args.example_metallic else examples_dir / "metallic.png"
    return example_rgb, example_metallic


def relpath_or_name(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def main() -> None:
    args = parse_args()
    ensure_api_key()

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

    variant_ids = resolve_variant_ids(args)
    input_dir = Path(args.input_dir).expanduser()
    seg_dir = Path(args.seg_dir).expanduser() if args.seg_dir else None
    example_rgb = None
    example_metallic = None

    if looks_like_object_seg_dir(input_dir):
        inferred_rgb_dir = infer_rgb_dir_from_seg_dir(input_dir)
        if inferred_rgb_dir is None:
            raise RuntimeError(
                f"input_dir looks like an ObjectSegmentation directory but paired RGB directory could not be inferred: {input_dir}"
            )
        print(f"[info] detected segmentation directory as input_dir, switching RGB directory to: {inferred_rgb_dir}")
        if seg_dir is None:
            seg_dir = input_dir
        input_dir = inferred_rgb_dir

    needs_seg = any(get_variant_config(variant_id).use_segmentation for variant_id in variant_ids)
    if needs_seg and seg_dir is None:
        seg_dir = infer_seg_dir_from_rgb_dir(input_dir)
        if seg_dir is not None:
            print(f"[info] inferred segmentation directory from input_dir: {seg_dir}")
    if needs_seg and seg_dir is None:
        raise ValueError("Variants a1/a2 require --seg_dir, or input_dir must have an inferable paired ObjectSegmentation directory.")
    if seg_dir is not None and (not seg_dir.exists() or not seg_dir.is_dir()):
        raise FileNotFoundError(f"seg_dir not found: {seg_dir}")

    needs_example = any(get_variant_config(variant_id).use_example_pair for variant_id in variant_ids)
    if needs_example:
        example_rgb, example_metallic = resolve_default_example_pair(args)
        ensure_example_pair(example_rgb, example_metallic)

    example_rgb_hash = file_sha256(example_rgb) if example_rgb else ""
    example_metallic_hash = file_sha256(example_metallic) if example_metallic else ""

    image_paths = filter_images_by_names(list_images(input_dir), args.include_names)
    if args.max_generate > 0:
        image_paths = image_paths[: args.max_generate]

    print(
        f"[1/3] found {len(image_paths)} RGB images; generating metallic maps with "
        f"model={args.image_model}, variants={','.join(variant_ids)}"
    )

    ark_client = Ark(base_url=args.base_url, api_key=ensure_api_key())
    ablation_root = Path(args.output_dir).expanduser()
    ablation_root.mkdir(parents=True, exist_ok=True)
    normalized_seg_suffix = normalize_seg_suffix(args.seg_suffix)

    print(f"[2/3] start generation for {len(image_paths)} images")
    for variant_id in variant_ids:
        config = get_variant_config(variant_id)
        signature_prompt = build_prompt(
            variant_id=variant_id,
            prompt_preset=args.prompt_preset,
        )
        run_signature = build_run_signature(
            variant_id=variant_id,
            image_model=args.image_model,
            prompt_text=signature_prompt,
            prompt_preset=args.prompt_preset,
            seg_dir=str(seg_dir) if seg_dir else "",
            seg_suffix=normalized_seg_suffix,
            example_rgb=str(example_rgb) if example_rgb else "",
            example_metallic=str(example_metallic) if example_metallic else "",
            example_rgb_sha256=example_rgb_hash,
            example_metallic_sha256=example_metallic_hash,
            timeout=args.timeout,
            size=args.size,
            watermark=args.watermark,
        )

        variant_dir = ablation_root / variant_id
        metallic_dir = variant_dir / "metallic"
        metallic_source_dir = variant_dir / "metallic_source"
        meta_dir = variant_dir / "meta"
        per_image_dir = meta_dir / "per_image"
        metallic_dir.mkdir(parents=True, exist_ok=True)
        per_image_dir.mkdir(parents=True, exist_ok=True)
        if config.postprocess_mode != "none":
            metallic_source_dir.mkdir(parents=True, exist_ok=True)

        setup = dict(run_signature)
        setup["repeat_index"] = args.repeat_index
        setup["num_images"] = len(image_paths)
        (meta_dir / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_path = meta_dir / "manifest.json"
        manifest = load_manifest(manifest_path)

        for idx, rgb_path in enumerate(image_paths, start=1):
            image_relpath = relpath_or_name(rgb_path, input_dir)
            metallic_relpath = image_relpath.with_name(f"{image_relpath.stem}_metallic.png")
            source_relpath = image_relpath.with_name(f"{image_relpath.stem}_metallic_source.png")

            item: Dict[str, Any] = {
                "image_name": rgb_path.name,
                "image_relpath": image_relpath.as_posix(),
                "variant_id": variant_id,
                "variant_name": config.variant_name,
                "repeat_index": args.repeat_index,
            }
            try:
                seg_path = None
                if config.use_segmentation:
                    if seg_dir is None:
                        raise ValueError(f"{variant_id} requires --seg_dir")
                    seg_path = find_matching_seg(rgb_path, seg_dir, preferred_suffix=args.seg_suffix)
                    item["seg_name"] = seg_path.name
                    print(f"  - [{variant_id}] ({idx}/{len(image_paths)}) {rgb_path.name} | seg={seg_path.name}")
                elif config.use_example_pair:
                    item["example_rgb"] = example_rgb.name if example_rgb else ""
                    item["example_metallic"] = example_metallic.name if example_metallic else ""
                    print(
                        f"  - [{variant_id}] ({idx}/{len(image_paths)}) {rgb_path.name} | "
                        f"example={example_rgb.name if example_rgb else ''}"
                    )
                else:
                    print(f"  - [{variant_id}] ({idx}/{len(image_paths)}) {rgb_path.name}")

                input_fingerprints: Dict[str, Any] = {
                    "rgb_sha256": file_sha256(rgb_path),
                    "seg_name": seg_path.name if seg_path else "",
                    "seg_sha256": file_sha256(seg_path) if seg_path else "",
                    "example_rgb_sha256": example_rgb_hash,
                    "example_metallic_sha256": example_metallic_hash,
                }

                metallic_path = metallic_dir / metallic_relpath
                source_path = metallic_source_dir / source_relpath if config.postprocess_mode != "none" else None
                output_meta_path = per_image_dir / metallic_relpath.with_suffix(".json")
                metallic_path.parent.mkdir(parents=True, exist_ok=True)
                output_meta_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path is not None:
                    source_path.parent.mkdir(parents=True, exist_ok=True)

                item["prompt_version"] = config.prompt_version
                item["input_mode"] = config.input_mode

                if args.skip_existing and should_skip_existing_output(
                    metallic_path=metallic_path,
                    output_meta_path=output_meta_path,
                    run_signature=run_signature,
                    input_fingerprints=input_fingerprints,
                ):
                    item["skipped"] = True
                    item["skip_reason"] = "matching_output_signature_and_input_fingerprints"
                    item["metallic_output"] = metallic_relpath.as_posix()
                else:
                    generated = generate_one_case(
                        ark_client=ark_client,
                        model=args.image_model,
                        variant_id=variant_id,
                        rgb_path=rgb_path,
                        seg_path=seg_path,
                        save_path=metallic_path,
                        example_rgb=example_rgb,
                        example_metallic=example_metallic,
                        size=args.size,
                        watermark=args.watermark,
                        timeout=args.timeout,
                        prompt_preset=args.prompt_preset,
                        source_prediction_path=source_path,
                    )
                    item["metallic_mode"] = generated["route"]
                    item["metallic_output"] = metallic_relpath.as_posix()
                    if generated.get("source_prediction"):
                        try:
                            item["source_prediction"] = Path(generated["source_prediction"]).relative_to(variant_dir).as_posix()
                        except ValueError:
                            item["source_prediction"] = generated["source_prediction"]

                    output_meta: Dict[str, Any] = {
                        "image_name": rgb_path.name,
                        "image_relpath": image_relpath.as_posix(),
                        "metallic_output": metallic_relpath.as_posix(),
                        "variant_id": variant_id,
                        "variant_name": config.variant_name,
                        "route": config.route,
                        "input_mode": config.input_mode,
                        "segmentation_role": config.segmentation_role,
                        "postprocess_mode": config.postprocess_mode,
                        "repeat_index": args.repeat_index,
                        "example_rgb": str(example_rgb) if example_rgb else "",
                        "example_metallic": str(example_metallic) if example_metallic else "",
                        "input_fingerprints": input_fingerprints,
                        "run_signature": run_signature,
                    }
                    if seg_path is not None:
                        output_meta["seg_name"] = seg_path.name
                    if generated.get("source_prediction"):
                        try:
                            output_meta["source_prediction"] = Path(generated["source_prediction"]).relative_to(variant_dir).as_posix()
                        except ValueError:
                            output_meta["source_prediction"] = generated["source_prediction"]
                    output_meta_path.write_text(
                        json.dumps(output_meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    time.sleep(max(0.0, args.sleep))
            except Exception as exc:
                item["error"] = str(exc)
                item["status"] = "error"
                print(f"[error] [{variant_id}] {rgb_path.name}: {exc}")

            upsert_manifest_entry(manifest, item)
            write_manifest(manifest_path, manifest)

    print("[3/3] done")
    print(f"output_dir: {ablation_root.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
