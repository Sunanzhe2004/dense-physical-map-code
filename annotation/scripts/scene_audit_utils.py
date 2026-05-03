"""Shared helpers for scene-level audit scripts."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
API_IMAGE_SUFFIX = "_im.png"
MAX_STAT_IMAGE_SIDE = 512
HIST_BINS = 1024
HIGHLIGHT_LIN_TH = 0.85
DARK_LIN_TH = 0.10
HIGHLIGHT_RATIO_LOW = 0.01
HIGHLIGHT_RATIO_HIGH = 0.05
DARK_RATIO_LOW = 0.10
DARK_RATIO_HIGH = 0.30
_PIL_WARNED = False


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _srgb_to_linear_lut() -> List[float]:
    lut = []
    for i in range(256):
        c = i / 255.0
        if c <= 0.04045:
            lut.append(c / 12.92)
        else:
            lut.append(((c + 0.055) / 1.055) ** 2.4)
    return lut


_SRGB_LUT = _srgb_to_linear_lut()


def _linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055


def _percentile_from_hist(hist: List[int], total: int, q: float) -> float:
    if total <= 0:
        return 0.0
    target = total * q
    cum = 0
    bins = len(hist)
    for i, count in enumerate(hist):
        cum += count
        if cum >= target:
            return i / (bins - 1)
    return 1.0


def bucket_brightness(mean_srgb: float) -> str:
    low_th = _linear_to_srgb(0.18 / 2.0)
    high_th = _linear_to_srgb(0.18 * 2.0)
    if mean_srgb < low_th:
        return "low"
    if mean_srgb > high_th:
        return "high"
    return "medium"


def bucket_illumination(mean_lin: float) -> str:
    eps = 1e-6
    ev = math.log2((mean_lin + eps) / 0.18)
    if ev <= -2.0:
        return "very_low"
    if ev <= -1.0:
        return "low"
    if ev <= 1.0:
        return "medium"
    if ev <= 2.0:
        return "high"
    return "very_high"


def bucket_dynamic_range(spread_stops: float) -> str:
    if spread_stops < 2.0:
        return "low"
    if spread_stops < 4.0:
        return "medium"
    return "high"


def bucket_ratio(ratio: float, low_th: float, high_th: float) -> str:
    if ratio < low_th:
        return "low"
    if ratio < high_th:
        return "medium"
    return "high"


def compute_lighting_stats(image_paths: List[str]) -> Optional[Dict[str, float]]:
    global _PIL_WARNED
    if Image is None:
        if not _PIL_WARNED:
            eprint("PIL is not installed; lighting statistics will be skipped.")
            _PIL_WARNED = True
        return None

    total_pixels = 0
    sum_lin = 0.0
    sum_srgb = 0.0
    highlight = 0
    dark = 0
    hist = [0] * HIST_BINS

    for path in image_paths:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                max_side = max(w, h)
                if max_side > MAX_STAT_IMAGE_SIDE:
                    scale = MAX_STAT_IMAGE_SIDE / float(max_side)
                    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
                for r, g, b in im.getdata():
                    y_srgb = 0.2126 * (r / 255.0) + 0.7152 * (g / 255.0) + 0.0722 * (b / 255.0)
                    y_lin = 0.2126 * _SRGB_LUT[r] + 0.7152 * _SRGB_LUT[g] + 0.0722 * _SRGB_LUT[b]
                    sum_srgb += y_srgb
                    sum_lin += y_lin
                    total_pixels += 1
                    if y_lin >= HIGHLIGHT_LIN_TH:
                        highlight += 1
                    if y_lin <= DARK_LIN_TH:
                        dark += 1
                    hist[int(y_lin * (HIST_BINS - 1))] += 1
        except Exception as exc:
            eprint(f"[{os.path.basename(path)}] failed to compute lighting statistics: {exc}")

    if total_pixels <= 0:
        return None

    mean_lin = sum_lin / total_pixels
    mean_srgb = sum_srgb / total_pixels
    p5 = _percentile_from_hist(hist, total_pixels, 0.05)
    p95 = _percentile_from_hist(hist, total_pixels, 0.95)
    spread_stops = math.log2((p95 + 1e-6) / (p5 + 1e-6))
    return {
        "mean_luminance": mean_lin,
        "mean_luma_srgb": mean_srgb,
        "p5_luminance": p5,
        "p95_luminance": p95,
        "dynamic_range_stops": spread_stops,
        "highlight_ratio": highlight / total_pixels,
        "dark_ratio": dark / total_pixels,
    }


def apply_lighting_buckets(parsed: Dict[str, Any], stats: Optional[Dict[str, float]]) -> None:
    if not stats:
        return
    parsed["brightness_level"] = bucket_brightness(stats["mean_luma_srgb"])
    parsed["illumination_level"] = bucket_illumination(stats["mean_luminance"])
    parsed["dynamic_range_level"] = bucket_dynamic_range(stats["dynamic_range_stops"])
    parsed["highlight_strength"] = bucket_ratio(stats["highlight_ratio"], HIGHLIGHT_RATIO_LOW, HIGHLIGHT_RATIO_HIGH)
    parsed["dark_region_ratio_level"] = bucket_ratio(stats["dark_ratio"], DARK_RATIO_LOW, DARK_RATIO_HIGH)


def list_scenes(input_dir: str) -> List[str]:
    return sorted(name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name)))


def list_images(scene_dir: str, suffix: str = API_IMAGE_SUFFIX) -> List[str]:
    images = []
    for name in os.listdir(scene_dir):
        path = os.path.join(scene_dir, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(path) and ext in IMAGE_EXTS and name.lower().endswith(suffix):
            images.append(path)
    return sorted(images)


def guess_mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "image/png"


def encode_image_data_url(path: str) -> str:
    with open(path, "rb") as handle:
        data = handle.read()
    return f"data:{guess_mime(path)};base64,{base64.b64encode(data).decode('ascii')}"


def extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "empty_response"
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None, "json_not_found"
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed, None
    except Exception as exc:
        return None, f"json_parse_error: {exc}"
    return None, "json_not_found"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def is_completed_scene(output_dir: str, scene: str) -> bool:
    scene_out = os.path.join(output_dir, f"{scene}.json")
    if not os.path.isfile(scene_out) or os.path.getsize(scene_out) <= 0:
        return False
    try:
        with open(scene_out, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("error")
