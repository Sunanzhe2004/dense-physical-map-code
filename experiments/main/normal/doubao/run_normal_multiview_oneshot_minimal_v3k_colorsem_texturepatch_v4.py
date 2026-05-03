#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ANALYSIS_MODEL = "doubao-seed-2-0-pro-260215"
DEFAULT_NORMAL_MODEL = "doubao-seedream-5-0-260128"
DEFAULT_EXAMPLE_RGB = "/path/to/benchmark_examples/normal/example1_rgb.png"
DEFAULT_EXAMPLE_NORMAL = "/path/to/benchmark_examples/normal/example1_normal.png"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-view normal map generation (one-shot minimal v3k_colorsem_texturepatch_v4; stronger top-facing calibration, flat-top stability, shading-to-bump veto, and tighter fabric simplification)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--filename_suffix", type=str, default="_im.png", help="Only process filenames ending with this suffix.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input_dir.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Ark base URL.")
    parser.add_argument(
        "--analysis_model",
        type=str,
        default=DEFAULT_ANALYSIS_MODEL,
        help="Analysis model (used in zero-shot mode).",
    )
    parser.add_argument("--normal_model", type=str, default=DEFAULT_NORMAL_MODEL, help="Normal model.")
    parser.add_argument(
        "--max_views",
        type=int,
        default=6,
        help="Max views for analysis (zero-shot only).",
    )
    parser.add_argument(
        "--analysis_max_side",
        type=int,
        default=1024,
        help="Resize long side before analysis.",
    )
    parser.add_argument(
        "--detail",
        type=str,
        default="high",
        choices=["low", "high", "auto"],
        help="Vision detail level.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--normal_size", type=str, default="adaptive", help="Normal output size.")
    parser.add_argument("--watermark", action="store_true", help="Keep watermark.")
    parser.add_argument("--max_generate", type=int, default=0, help="Max images to generate (0 = all).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs and analysis files.",
    )
    parser.add_argument(
        "--independent_images",
        action="store_true",
        help="Treat each image as an independent sample and analyze/generate it separately.",
    )
    parser.add_argument(
        "--preserve_relative_dirs",
        action="store_true",
        help="Preserve relative directory layout under input_dir.",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests.")
    parser.add_argument("--timeout", type=int, default=120, help="Download timeout in seconds.")
    parser.add_argument(
        "--example_rgb",
        type=str,
        default=DEFAULT_EXAMPLE_RGB,
        help="One-shot example RGB path. Default enables one-shot mode.",
    )
    parser.add_argument(
        "--example_normal",
        type=str,
        default=DEFAULT_EXAMPLE_NORMAL,
        help="One-shot example normal path. Default enables one-shot mode.",
    )
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY in the environment.")
    return api_key



def list_images(input_dir: Path, filename_suffix: Optional[str] = None, recursive: bool = False) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if filename_suffix:
        images = [p for p in images if p.name.endswith(filename_suffix)]
    if not images:
        suffix_msg = f" ending with {filename_suffix}" if filename_suffix else ""
        recursive_msg = " (recursive scan)" if recursive else " (top-level scan only)"
        raise FileNotFoundError(
            f"No supported image files found{suffix_msg}{recursive_msg}: {input_dir}"
        )
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


def pil_resize_bytes(path: Path, max_side: int) -> Tuple[bytes, str]:
    if Image is None:
        data = path.read_bytes()
        return data, guess_mime(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / float(long_side)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


def file_to_data_uri(path: Path, max_side: Optional[int] = None) -> str:
    if max_side is not None:
        data, mime = pil_resize_bytes(path, max_side=max_side)
    else:
        data = path.read_bytes()
        mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def clean_json_text(text: str) -> str:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    return text


def robust_json_loads(text: str) -> Dict[str, Any]:
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_completed_output(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = load_json_file(path)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def upsert_manifest_entry(manifest: List[Dict[str, Any]], entry: Dict[str, Any], key_field: str = "relative_image_path") -> None:
    entry_key = entry.get(key_field)
    if entry_key is None:
        manifest.append(entry)
        return
    for idx, item in enumerate(manifest):
        if item.get(key_field) == entry_key:
            manifest[idx] = entry
            return
    manifest.append(entry)


def extract_error_code(exc: Exception) -> str:
    text = str(exc)
    patterns = [
        r"'code':\s*'([^']+)'",
        r'"code":\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return exc.__class__.__name__


def should_skip_image_error(exc: Exception) -> bool:
    text = str(exc)
    skip_patterns = [
        "InputTextSensitiveContentDetected",
        "InputImageSensitiveContentDetected",
        "SensitiveContentDetected",
        "sensitive information",
        "sensitive content",
    ]
    return any(pattern in text for pattern in skip_patterns)


def build_skip_path(meta_dir: Path) -> Path:
    return meta_dir / "skipped_images.json"


def record_skipped_image(
    skipped_images: List[Dict[str, Any]],
    skipped_path: Path,
    image_path: Path,
    input_dir: Path,
    stage: str,
    exc: Exception,
) -> None:
    entry = {
        "skip_key": f"{image_path.relative_to(input_dir).as_posix()}::{stage}",
        "image_name": image_path.name,
        "relative_image_path": image_path.relative_to(input_dir).as_posix(),
        "stage": stage,
        "error_code": extract_error_code(exc),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "skipped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    upsert_manifest_entry(skipped_images, entry, key_field="skip_key")
    skipped_path.write_text(json.dumps(skipped_images, ensure_ascii=False, indent=2), encoding="utf-8")


def was_image_skipped(skipped_images: List[Dict[str, Any]], image_path: Path, input_dir: Path) -> Optional[Dict[str, Any]]:
    relative_path = image_path.relative_to(input_dir).as_posix()
    for item in skipped_images:
        if item.get("relative_image_path") == relative_path:
            return item
    return None


def extract_output_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return output_text

    if hasattr(resp, "output"):
        texts: List[str] = []
        for item in resp.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                txt = getattr(c, "text", None)
                if txt:
                    texts.append(txt)
        if texts:
            return "\n".join(texts)

    choices = getattr(resp, "choices", None)
    if choices:
        texts: List[str] = []
        for choice in choices:
            message = getattr(choice, "message", None)
            if not message:
                continue
            content = getattr(message, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                texts.append(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        texts.append(part["text"])
                    else:
                        txt = getattr(part, "text", None)
                        if txt:
                            texts.append(txt)
        if texts:
            return "\n".join(texts)

    return str(resp)


def save_url_to_file(url: str, save_path: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
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
    raise RuntimeError("Response has neither url nor b64_json; cannot save image.")


def build_analysis_prompt(image_names: List[str]) -> str:
    names_text = "\n".join(f"- {name}" for name in image_names)
    return f"""
You will be shown multiple RGB views of the same indoor scene. Your task is not to generate an image. Instead, provide a very short geometric summary that can support zero-shot normal-map generation.

Requirements:
1. Focus only on large-scale geometry, not texture, shadows, highlights, or color.
2. If a detail looks more like material texture than true shape, ignore it.
3. Keep the summary concise and retain only stable structural information.
4. Output JSON only, with no extra explanation.

Input view filenames:
{names_text}

Return exactly this JSON structure:
{{
  "scene_summary": "one-sentence scene summary",
  "geometry_notes": ["2 to 6 large-scale geometry notes"],
  "per_view": [
    {{
      "image_name": "must exactly match an input filename",
      "normal_hint": "one short geometry note for this view"
    }}
  ]
}}
""".strip()

def analyze_multiview(
    client: OpenAI,
    image_paths: List[Path],
    model: str,
    detail: str = "high",
    analysis_max_side: int = 1024,
) -> Dict[str, Any]:
    image_names = [p.name for p in image_paths]
    prompt = build_analysis_prompt(image_names)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        image_url = {"url": file_to_data_uri(path, max_side=analysis_max_side)}
        if detail in {"low", "high"}:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
    )
    text = extract_output_text(resp)
    data = robust_json_loads(text)
    data.setdefault("scene_summary", "")
    data.setdefault("geometry_notes", [])
    data.setdefault("per_view", [])
    return data


def get_per_view_hints(global_analysis: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in global_analysis.get("per_view", []):
        image_name = item.get("image_name")
        if image_name:
            result[image_name] = item.get("normal_hint", "")
    return result


def make_zero_shot_prompt(scene_summary: str, geometry_notes: List[str], per_view_hint: str) -> str:
    notes = "; ".join(geometry_notes) if geometry_notes else "N/A"
    return (
        "Generate a strict view-space per-pixel surface normal map for an indoor scene. "
        "This is a dense geometric normal map, not an artistic rendering and not an edge visualization. "
        "Use RGB to encode normal direction only. "
        "Do not preserve albedo color, shading, highlights, reflections, wood grain, fabric color, wall color, floor color, or lighting tint. "
        "Suppress material texture and high-frequency appearance detail. "
        "If a detail looks like texture rather than true geometry, remove it. "
        "Preserve only object boundaries, large-scale geometry, broad curvature, panel recesses, frame thickness, and real shape transitions. "
        "Large planar or gently curved regions should be smooth and nearly texture-free. "
        "Visible door faces, window frames, sills, recess walls, sofa surfaces, and table surfaces are valid surfaces, not holes. "
        "Do not invent fake grooves, texture relief, or neon contour lines. "
        "Output a clean, pixel-aligned indoor normal map.\n\n"
        f"Scene summary: {scene_summary or 'N/A'}\n"
        f"Geometry notes: {notes}\n"
        f"View-specific hint: {per_view_hint or 'N/A'}"
    )


def make_one_shot_prompt() -> str:
    return (
        "You are given three images in order: "
        "(1) an example indoor RGB image, "
        "(2) the target normal map for that example, "
        "(3) the query indoor RGB image to convert. "

        "Generate the normal map for image (3). "
        "Image (2) is the only convention reference and has the highest priority, "
        "but copy only its stable encoding convention, not its local texture-like pixel patterns or residual noise. "
        "Use image (2) only for palette convention, direction-family convention, surface-role convention, deeper-color convention, boundary style, saturation range, and large-region consistency. "

        "Critical rules: "
        "color must encode surface orientation only, never material color, never albedo, never lighting tint. "
        "Do not preserve wood color, fabric color, wall color, floor color, or object appearance from the query RGB image. "
        "Do not preserve wood grain, fabric weave, printed patterns, shading, highlights, reflections, image noise, or any illumination residue. "

        "Normal-map knowledge: "
        "a normal map is not a natural color photo. It is directional data. "
        "RGB channels encode the 3D surface normal vector after image-space encoding. "
        "Treat every color as a direction family, not as an object class and not as a material class. "
        "Similar colors mean similar surface directions. Smooth color gradients mean smooth changes of surface orientation. "
        "Abrupt color changes should appear mainly at real geometric boundaries, sharp folds, depth discontinuities, object boundaries, or panel recesses. "
        "Large flat regions with the same orientation should have nearly constant color. "

        "Color-meaning guide: "
        "in many normal-map encodings, the red channel mainly responds to left-versus-right tilt, the green channel mainly responds to up-versus-down tilt, and the blue channel is strongest when a surface faces the camera more directly. "
        "A neutral front-facing normal is often close to a blue-dominant color family, while opposite tilts move the red or green channels in opposite directions. "
        "However, the exact sign and palette are convention-dependent, and the green channel may be flipped in different pipelines. "
        "Therefore do not impose a universal textbook palette. "
        "Infer the exact color-family meaning from image (2), then apply that convention consistently to image (3). "
        "If image (2) uses a blue-violet family for front-facing walls, then analogous front-facing walls in image (3) should use that family. "
        "If image (2) uses a cyan or green-shifted family for upward-facing horizontal tops, then analogous tops in image (3) should use that family. "
        "If image (2) uses different magenta, purple, cyan, or blue families for opposite side tilts or recess faces, preserve those differences only when the structural role is analogous. "
        "Do not make two regions different colors unless their directions or structural roles are truly different under the reference convention. "

        "Top-facing calibration rule: "
        "Visible upward-facing horizontal planes such as desktop tops, shelf top ledges, broad bed top regions, cabinet tops, and similar horizontal support surfaces must follow the upward-facing family of the reference normal map. "
        "If the reference target normal encodes upward-facing planes as greener or green-cyan compared with front-facing walls, keep that separation in image (3). "
        "Do not let front-facing wall families leak onto horizontal tops, and do not recolor horizontal tops based on material color or brightness. "

        "Orientation separation rule: "
        "Distinguish clearly between front-facing vertical faces, upward-facing horizontal faces, downward-facing undersides, and side-facing faces. "
        "Adjacent orthogonal faces should use distinct reference families when the reference normal map separates them. "

        "Palette calibration rule: "
        "match the overall chroma, saturation, and family spacing of image (2). "
        "Do not exaggerate cyan-magenta contrast and do not make the result look like a neon artistic rendering. "
        "If image (2) uses muted lavender, muted blue, muted cyan, or muted magenta families, keep those families similarly muted in image (3). "
        "Do not push walls, ceilings, or broad planes to a darker or more saturated family than the analogous reference region unless the direction truly changes. "
        "For broad planar regions, prefer stable muted reference families over dramatic color contrast. "

        "Saturation clamp rule: "
        "never make the output more saturated, more contrasty, or more neon-like than the reference target normal. "
        "If uncertain between two plausible families, choose the slightly more muted and more stable one. "
        "Prefer muted dataset-like normal colors over vivid artistic cyan-magenta contrast. "

        "Planar family consistency rule: "
        "for one continuous planar region with nearly constant orientation, use one stable reference family across the whole region. "
        "Do not introduce large blue-purple or cyan-magenta drift across the same wall, same cabinet face, same desktop, same drawer front, same shelf face, or same curtain panel when the orientation is nearly constant. "
        "Brightness, shadow, occlusion, distance, and local contrast in the RGB image must not split one planar region into multiple normal families. "

        "Lighting-removal rule: "
        "ignore illumination completely. "
        "Do not encode lamp glow, ceiling-light hotspots, window brightening, cast shadows, soft shadow gradients, interreflection, ambient occlusion-like darkening, exposure falloff, bloom, glare, or colored light spill into the normal map. "
        "A region must not become brighter, darker, more purple, or more cyan merely because it is more illuminated or more shadowed in the RGB image. "
        "If two visible surfaces have the same geometric orientation but different brightness because of lighting, they should still stay in the same normal-map family. "
        "Darkness caused by lighting is not a hole and is not a geometry cue. "
        "Only true geometric orientation and the reference convention may determine output color. "
        "For visible light fixtures, preserve only the fixture silhouette or boundary if clearly visible; do not preserve any surrounding glow halo or light spill on nearby surfaces. "

        "No ambient-occlusion rule: "
        "Do not darken or recolor a region merely because it is recessed, near a corner, under a shelf, behind an object, near the floor, or inside a local cavity. "
        "Contact shadow, ambient occlusion, cavity darkening, and corner darkening are not normal signals. "
        "Use a different family only when the visible face orientation truly changes, not merely because the region is more enclosed or less illuminated. "

        "Flat-top stability rule: "
        "Desktop tops, shelf tops, cabinet tops, broad bed top regions, and other large upward-facing planes should be very smooth and nearly uniform in one stable upward-facing family. "
        "Do not introduce blue-purple drift, local cyan patches, or small orientation ripples on such planes unless there is a true large-scale fold or curvature change. "

        "Shading-to-bump veto rule: "
        "Do not infer extra normal detail from soft shading, local brightness streaks, blanket lighting bands, curtain brightness variation, material gloss, or photometric contrast alone. "
        "Photometric contrast without a stable geometric boundary is not geometry and must not become bumps, grooves, or repeated surface ripples. "

        "Very strong texture suppression rule: "
        "remove fine-scale and mid-scale material texture. "
        "Do not transfer wood grain, cloth weave, subtle bumps from fabric, or streak-like appearance patterns into the normal map. "
        "If a local variation is not clearly a true shape change, treat it as texture and suppress it. "
        "Large door faces, table tops, sofa backs, sofa seats, sofa arms, walls, and other broad surfaces should be smooth, low-variance, and nearly texture-free. "
        "Even if image (2) contains faint texture-like residuals, do not copy them literally. Prefer a smoother result. "

        "Extra texture veto rule: "
        "Printed bedding patterns, plaid, stripes, checkers, quilt seams, cloth weave, fabric print, repeated textile motifs, logos, labels, and tiny decorative appearance cues are not geometry. "
        "Suppress them even when they are high-contrast. "
        "On beds, blankets, pillows, curtains, sofas, and upholstered furniture, preserve only low-frequency folds, broad drape, smooth inflation, seam-level macro shape only when it is clearly geometric, and true silhouette changes. "
        "Do not convert textile pattern contrast, woven structure, printed motifs, fine stitch lines, or hem shading into normal variation. "
        "For small objects such as bottles, books, cups, phones, remotes, and plant pots, preserve only coarse shape, main visible faces, and broad curvature. "
        "Simplify away tiny appearance details, printed markings, logos, color contrast, and object-specific surface texture. "

        "Soft-fabric macro-shape rule: "
        "for curtains, blankets, bedsheets, pillows, sofas, and other soft fabric surfaces, preserve only broad folds, major crests, major valleys, drape direction, and smooth inflation. "
        "Do not generate repeated fine ribbing, quilting bands, stitch-like grooves, woven texture, cloth pattern, grid lines, check patterns, or evenly spaced micro-waves unless they are clearly dominant large-scale geometric folds in the reference target normal. "
        "On blankets and bedsheets, collapse weak internal cloth variation into a smoother broad surface and keep only a small number of dominant folds. "
        "If uncertain, simplify fabric to fewer, broader folds. "

        "Reference analogy rule: "
        "for each large visible region in image (3), internally match it to the closest analogous region in image (2) by structural role and direction. "
        "Examples include flat wall patch, curtain outer fold, curtain inner groove, shelf outer face, shelf inner cavity wall, desktop top, drawer front, bed top surface, bed side surface, chair outer shell, chair inner seat, window frame, sill, recess wall, and opening. "
        "Reuse the corresponding family from image (2) for that analogous region. "
        "Do not merge distinct reference families just because two query regions have similar brightness. "

        "Refined deeper-color rule: "
        "do not use a deeper family merely because a region is farther, more recessed, partially occluded, or darker in the RGB image. "
        "Use a deeper family only when the analogous region in the reference target normal clearly belongs to a different visible orientation family, or when it is a true opening or void-like region. "
        "Shallow recesses and enclosed visible surfaces should still follow the correct visible-face family from the reference. "
        "Use pure black only where image (2) clearly uses pure black for openings, masked regions, or void-like regions. "
        "Do not expand black to cover ordinary visible structures. "
        "Window frames, mullions, sills, recess walls, door faces, desk faces, and other visible solid surfaces must remain in a valid visible-surface family. "
        "Preserve visible frame structure around openings. "

        "Geometry rule: "
        "preserve the query image layout, object boundaries, major silhouettes, broad curvature, curtain folds, blanket folds, panel recesses, frame thickness, and true shape discontinuities. "
        "Use narrow transitions only at true geometric boundaries. "
        "Do not invent fake bevels, fake grooves, fake seams, texture relief, or artistic contour lines. "

        "Priority order: "
        "1) preserve query geometry and layout, "
        "2) infer the meaning of each reference color family from image (2), "
        "3) match the saturation and family spacing of image (2), "
        "4) keep one stable family for one nearly planar region, "
        "5) map each query region to the closest analogous reference region, "
        "6) remove all lighting and illumination effects, "
        "7) suppress texture aggressively, "
        "8) never copy material color or material texture from image (3). "

        "Output only the final normal map for image (3)."
    )


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
    image_uris = [file_to_data_uri(p, max_side=None) for p in image_paths]
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

    try:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris if len(image_uris) > 1 else image_uris[0],
            seed=seed,
            guidance_scale=guidance_scale,
            size=size,
            watermark=watermark,
        )
    except TypeError:
        return ark_client.images.generate(
            model=model,
            prompt=prompt,
            image=image_uris[-1],
            size=size,
            watermark=watermark,
        )


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
            image_path=image_path,
            input_dir=input_dir,
            base_dir=normal_dir,
            suffix="_normal.png",
            preserve_relative_dirs=preserve_relative_dirs,
        )
        if is_completed_output(out_normal):
            continue
        pending.append(image_path)
    return pending


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key()

    if Ark is None:
        raise ImportError(
            'Failed to import volcenginesdkarkruntime. Please install: pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR

    example_rgb = Path(args.example_rgb) if args.example_rgb else None
    example_normal = Path(args.example_normal) if args.example_normal else None
    use_oneshot = bool(example_rgb and example_normal)

    if (example_rgb and not example_normal) or (example_normal and not example_rgb):
        raise ValueError("One-shot mode requires both --example_rgb and --example_normal.")

    if use_oneshot:
        if not example_rgb.exists():
            raise FileNotFoundError(f"example_rgb not found: {example_rgb}")
        if not example_normal.exists():
            raise FileNotFoundError(f"example_normal not found: {example_normal}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    normal_dir = output_dir / "normal"
    meta_dir = output_dir / "meta"
    normal_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir, args.filename_suffix, args.recursive)
    image_paths_for_generate = image_paths[: args.max_generate] if args.max_generate > 0 else image_paths
    pending_image_paths = get_pending_images(
        image_paths=image_paths_for_generate,
        input_dir=input_dir,
        normal_dir=normal_dir,
        overwrite=args.overwrite,
        preserve_relative_dirs=args.preserve_relative_dirs,
    )

    print(
        f"[1/3] Found {len(image_paths)} images; candidate generations: {len(image_paths_for_generate)}; "
        f"pending: {len(pending_image_paths)}; one_shot={use_oneshot}"
    )
    print(
        "      "
        f"filename_suffix={args.filename_suffix or 'N/A'} | "
        f"recursive={args.recursive} | "
        f"preserve_relative_dirs={args.preserve_relative_dirs} | "
        f"independent_images={args.independent_images} | "
        f"overwrite={args.overwrite}"
    )

    oa_client = OpenAI(base_url=args.base_url, api_key=api_key)
    ark_client = Ark(base_url=args.base_url, api_key=api_key)

    manifest_path = meta_dir / "manifest.json"
    skipped_path = build_skip_path(meta_dir)
    manifest = load_manifest(manifest_path)
    skipped_images = load_manifest(skipped_path)

    if args.independent_images:
        analysis_dir = meta_dir / "per_image_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        print("[2/3] Independent-image mode: process images one by one without shared multi-view analysis.")
        print(f"[3/3] Start exporting normal maps one by one (pending: {len(pending_image_paths)} images)")
        if not pending_image_paths:
            print("  - The current directory is already complete; exiting.")
        for idx, image_path in enumerate(image_paths_for_generate, start=1):
            out_normal = build_image_output_path(
                image_path=image_path,
                input_dir=input_dir,
                base_dir=normal_dir,
                suffix="_normal.png",
                preserve_relative_dirs=args.preserve_relative_dirs,
            )
            single_analysis_path = build_image_output_path(
                image_path=image_path,
                input_dir=input_dir,
                base_dir=analysis_dir,
                suffix=".json",
                preserve_relative_dirs=args.preserve_relative_dirs,
            )
            skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
            if skipped_entry is not None:
                print(
                    f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} "
                    f"previously skipped, reason={skipped_entry.get('error_code', 'N/A')}"
                )
                continue
            if not args.overwrite and is_completed_output(out_normal):
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} already exists; skipping")
                continue
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")

            if use_oneshot:
                normal_prompt = make_one_shot_prompt()
                normal_input_paths = [example_rgb, example_normal, image_path]
                single_analysis: Dict[str, Any] = {
                    "mode": "one_shot_v3k_colorsem_texturepatch_v4",
                    "analysis_skipped": True,
                    "example_rgb": str(example_rgb),
                    "example_normal": str(example_normal),
                }
                if args.overwrite or not single_analysis_path.exists():
                    ensure_parent_dir(single_analysis_path)
                    single_analysis_path.write_text(
                        json.dumps(single_analysis, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            else:
                if not args.overwrite and single_analysis_path.exists():
                    single_analysis = load_json_file(single_analysis_path)
                    print("      Reusing existing per-image analysis")
                else:
                    try:
                        single_analysis = analyze_multiview(
                            client=oa_client,
                            image_paths=[image_path],
                            model=args.analysis_model,
                            detail=args.detail,
                            analysis_max_side=args.analysis_max_side,
                        )
                    except Exception as exc:
                        if should_skip_image_error(exc):
                            print(f"      Skip: analysis stage triggered a skippable error -> {extract_error_code(exc)}")
                            record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "analysis", exc)
                            continue
                        raise
                    ensure_parent_dir(single_analysis_path)
                    single_analysis_path.write_text(
                        json.dumps(single_analysis, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                per_view_hints = get_per_view_hints(single_analysis)
                normal_prompt = make_zero_shot_prompt(
                    scene_summary=single_analysis.get("scene_summary", ""),
                    geometry_notes=single_analysis.get("geometry_notes", []),
                    per_view_hint=per_view_hints.get(image_path.name, ""),
                )
                normal_input_paths = [image_path]

            try:
                normal_resp = run_image_generation(
                    ark_client=ark_client,
                    model=args.normal_model,
                    prompt=normal_prompt,
                    image_paths=normal_input_paths,
                    size=args.normal_size,
                    watermark=args.watermark,
                    seed=args.seed,
                    guidance_scale=args.guidance_scale,
                )

                ensure_parent_dir(out_normal)
                save_image_response(normal_resp.data[0], out_normal, timeout=args.timeout)
            except Exception as exc:
                if should_skip_image_error(exc):
                    print(f"      Skip: generation stage triggered a skippable error -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc)
                    continue
                raise
            upsert_manifest_entry(
                manifest,
                {
                    "image_name": image_path.name,
                    "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                    "analysis_model": args.analysis_model if not use_oneshot else "",
                    "normal_model": args.normal_model,
                    "mode": "one_shot_v3k_colorsem_texturepatch_v4" if use_oneshot else "zero_shot_single_image",
                    "example_rgb": str(example_rgb) if example_rgb else "",
                    "example_normal": str(example_normal) if example_normal else "",
                    "normal_output": out_normal.relative_to(output_dir).as_posix(),
                    "analysis_output": single_analysis_path.relative_to(output_dir).as_posix(),
                    "normal_prompt": normal_prompt,
                },
            )
            time.sleep(max(0.0, args.sleep))
    else:
        image_paths_for_analysis = image_paths[: max(1, args.max_views)]
        global_analysis: Dict[str, Any] = {}
        per_view_hints: Dict[str, str] = {}

        if not pending_image_paths:
            print("[2/3] The current directory is already complete; skipping analysis and generation.")
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("Done.")
            print(f"Normal output dir: {normal_dir.as_posix()}")
            print(f"Meta output dir: {meta_dir.as_posix()}")
            return

        if use_oneshot:
            meta_info = {
                "mode": "one_shot_v3k_colorsem_texturepatch_v4",
                "analysis_skipped": True,
                "example_rgb": str(example_rgb),
                "example_normal": str(example_normal),
            }
            (meta_dir / "multiview_analysis.json").write_text(
                json.dumps(meta_info, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("[2/3] One-shot mode: global analysis skipped.")
        else:
            global_analysis = analyze_multiview(
                client=oa_client,
                image_paths=image_paths_for_analysis,
                model=args.analysis_model,
                detail=args.detail,
                analysis_max_side=args.analysis_max_side,
            )
            per_view_hints = get_per_view_hints(global_analysis)
            (meta_dir / "multiview_analysis.json").write_text(
                json.dumps(global_analysis, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[2/3] Zero-shot mode: multi-view analysis completed and saved to: {(meta_dir / 'multiview_analysis.json').as_posix()}")

        print(f"[3/3] Start exporting normal maps ({len(image_paths_for_generate)} images)")
        for idx, image_path in enumerate(image_paths_for_generate, start=1):
            out_normal = build_image_output_path(
                image_path=image_path,
                input_dir=input_dir,
                base_dir=normal_dir,
                suffix="_normal.png",
                preserve_relative_dirs=args.preserve_relative_dirs,
            )
            skipped_entry = None if args.overwrite else was_image_skipped(skipped_images, image_path, input_dir)
            if skipped_entry is not None:
                print(
                    f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} "
                    f"previously skipped, reason={skipped_entry.get('error_code', 'N/A')}"
                )
                continue
            if not args.overwrite and is_completed_output(out_normal):
                print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name} already exists; skipping")
                continue
            print(f"  - ({idx}/{len(image_paths_for_generate)}) {image_path.name}")

            if use_oneshot:
                normal_prompt = make_one_shot_prompt()
                normal_input_paths = [example_rgb, example_normal, image_path]
            else:
                normal_prompt = make_zero_shot_prompt(
                    scene_summary=global_analysis.get("scene_summary", ""),
                    geometry_notes=global_analysis.get("geometry_notes", []),
                    per_view_hint=per_view_hints.get(image_path.name, ""),
                )
                normal_input_paths = [image_path]

            try:
                normal_resp = run_image_generation(
                    ark_client=ark_client,
                    model=args.normal_model,
                    prompt=normal_prompt,
                    image_paths=normal_input_paths,
                    size=args.normal_size,
                    watermark=args.watermark,
                    seed=args.seed,
                    guidance_scale=args.guidance_scale,
                )

                ensure_parent_dir(out_normal)
                save_image_response(normal_resp.data[0], out_normal, timeout=args.timeout)
            except Exception as exc:
                if should_skip_image_error(exc):
                    print(f"      Skip: generation stage triggered a skippable error -> {extract_error_code(exc)}")
                    record_skipped_image(skipped_images, skipped_path, image_path, input_dir, "generation", exc)
                    continue
                raise
            upsert_manifest_entry(
                manifest,
                {
                    "image_name": image_path.name,
                    "relative_image_path": image_path.relative_to(input_dir).as_posix(),
                    "analysis_model": args.analysis_model if not use_oneshot else "",
                    "normal_model": args.normal_model,
                    "mode": "one_shot_v3k_colorsem_texturepatch_v4" if use_oneshot else "zero_shot",
                    "example_rgb": str(example_rgb) if example_rgb else "",
                    "example_normal": str(example_normal) if example_normal else "",
                    "normal_output": out_normal.relative_to(output_dir).as_posix(),
                    "normal_prompt": normal_prompt,
                },
            )
            time.sleep(max(0.0, args.sleep))

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Normal output dir: {normal_dir.as_posix()}")
    print(f"Meta output dir: {meta_dir.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nInterrupted.", file=sys.stderr)
        sys.exit(130)
