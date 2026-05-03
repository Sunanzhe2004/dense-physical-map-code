#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Final Doubao metallic generation script fixed to the gate_black prompt variant.

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
PROMPT_FAMILY = "metallic_rgb_prompt"
INPUT_MODE_CHOICES = ("rgb_only", "rgb_plus_prompt", "rgb_plus_example")
PROMPT_PRESET_CHOICES = ("v3_visualprior_noboundary",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metallic maps with SeedDream using the final gate_black prompt."
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
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Ark base URL")
    parser.add_argument("--image_model", type=str, default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument("--size", type=str, default="adaptive", help="Output size; adaptive -> 2k")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request and download timeout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--max_generate", type=int, default=0, help="0 means process all images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist")
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
    watermark: bool,
) -> Dict[str, Any]:
    if input_mode == "rgb_plus_prompt":
        route = "seedream_rgb_prompt_metallic"
    elif input_mode == "rgb_plus_example":
        route = "seedream_rgb_example_metallic"
    else:
        route = "seedream_rgb_only_metallic"

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
        "output_resolution_policy": "model_native_output",
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


def generate_metallic_map_with_seedream(
    *,
    ark_client: Any,
    model: str,
    input_mode: str,
    prompt_preset: str,
    rgb_path: Path,
    save_path: Path,
    scene_prompt: str,
    example_rgb: Optional[Path],
    example_metallic: Optional[Path],
    size: str,
    watermark: bool,
    timeout: int,
) -> str:
    prompt = build_metallic_prompt(
        input_mode=input_mode,
        prompt_preset=prompt_preset,
        scene_prompt=scene_prompt,
    )

    if input_mode == "rgb_plus_example":
        if not example_rgb or not example_metallic:
            raise ValueError("rgb_plus_example requires both --example_rgb and --example_metallic")
        image_paths = [example_rgb, example_metallic, rgb_path]
        mode = "seedream_rgb_example_metallic"
    elif input_mode == "rgb_plus_prompt":
        image_paths = [rgb_path]
        mode = "seedream_rgb_prompt_metallic"
    else:
        image_paths = [rgb_path]
        mode = "seedream_rgb_only_metallic"

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
    api_key = ensure_api_key()
    prompt_version = f"{PROMPT_FAMILY}_{args.prompt_preset}"

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

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
        f"[1/3] found {len(image_paths)} RGB images; generating {len(image_paths_for_generate)} metallic maps with model={args.image_model}, input_mode={args.input_mode}"
    )

    ark_client = Ark(base_url=args.base_url, api_key=api_key)
    prompt_text = build_metallic_prompt(
        args.input_mode,
        prompt_preset=args.prompt_preset,
        scene_prompt="[per-image prompt inserted at runtime]" if args.input_mode == "rgb_plus_prompt" else "",
    )
    run_signature = build_run_signature(
        input_mode=args.input_mode,
        prompt_version=prompt_version,
        prompt_preset=args.prompt_preset,
        image_model=args.image_model,
        prompt_text=prompt_text,
        prompt_dir=str(prompt_dir) if prompt_dir else "",
        prompt_suffix=args.prompt_suffix,
        example_rgb=str(example_rgb) if example_rgb else "",
        example_metallic=str(example_metallic) if example_metallic else "",
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
                item["metallic_mode"] = generate_metallic_map_with_seedream(
                    ark_client=ark_client,
                    model=args.image_model,
                    input_mode=args.input_mode,
                    prompt_preset=args.prompt_preset,
                    rgb_path=rgb_path,
                    save_path=metallic_path,
                    scene_prompt=scene_prompt,
                    example_rgb=example_rgb,
                    example_metallic=example_metallic,
                    size=args.size,
                    watermark=args.watermark,
                    timeout=args.timeout,
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
