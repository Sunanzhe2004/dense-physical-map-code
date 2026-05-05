#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import OpenEXR
    import Imath
except Exception:
    OpenEXR = None
    Imath = None

try:
    from scipy.stats import kendalltau as scipy_kendalltau
except Exception:
    scipy_kendalltau = None


DEFAULT_GT_ROOT = Path("data/depth/gt")
DEFAULT_PRED_ROOT = Path("outputs/evaluation/depth/predictions")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation/depth/results")
GT_DEPTH_SUFFIXES = ("_depth_raw.exr", "_depth_raw.dat", "_depth.exr", "_depth.dat")
PRED_DEPTH_SUFFIXES = (
    "_im_relative_depth.npy",
    "_relative_depth.npy",
    "_depth_raw.npy",
    "_depth.npy",
    "_im.npy",
    "_im_relative_depth.png",
    "_relative_depth.png",
    "_depth_raw.png",
    "_depth.png",
    "_im.png",
    ".png",
)
RGB_SUFFIX = "_im.png"
MASK_SUFFIX = "_mask.png"


@dataclass
class PairInfo:
    name: str
    category: str
    scene: str
    frame_id: str
    gt_raw_path: Path
    gt_rgb_path: Optional[Path]
    gt_mask_path: Optional[Path]
    pred_path: Path


@dataclass
class ImageMetrics:
    name: str
    category: str
    scene: str
    frame_id: str
    gt_raw_path: str
    gt_rgb_path: str
    gt_mask_path: str
    pred_path: str
    source_format: str
    width: int
    height: int
    pred_width: int
    pred_height: int
    resized_pred: bool
    resize_method: str
    pred_polarity: str
    valid_pixels: int
    mask_used: bool
    mask_pixels: int
    mask_excluded_pixels: int
    affine_scale: float
    affine_shift: float
    spearman_rho: float
    kendall_tau: float
    absrel_ai: float
    rmse_ai: float
    rms_ai: float
    mae_ai: float
    delta1_ai: float
    delta2_ai: float
    delta3_ai: float
    delta05_ai: float
    log10_ai: float
    edge_detail_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate corresponding depth prediction files under a GT directory and a prediction directory. "
            "GT supports *_depth.exr, *_depth_raw.exr, *_depth.dat, and *_depth_raw.dat."
        )
    )
    parser.add_argument("--gt_root", type=Path, default=DEFAULT_GT_ROOT, help=f"GT root. Default: {DEFAULT_GT_ROOT}")
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=None,
        help=(
            "Optional mask root. If omitted, masks are searched next to each GT depth file. "
            "Mask filenames are expected to be <frame_id>_mask.png."
        ),
    )
    parser.add_argument(
        "--pred_root",
        type=Path,
        default=DEFAULT_PRED_ROOT,
        help=f"Prediction root with corresponding files. Default: {DEFAULT_PRED_ROOT}",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save CSV/JSON summaries. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--gt_format",
        type=str,
        default="auto",
        choices=["auto", "dat", "exr"],
        help="GT depth format to use. 'auto' prefers DAT when both DAT and EXR exist. Default: auto.",
    )
    parser.add_argument(
        "--include_category_contains",
        type=str,
        default="",
        help=(
            "Optional comma-separated category-name substrings to evaluate, e.g. "
            "'mainaxis' or 'stresstest'. Empty means evaluate all categories."
        ),
    )
    parser.add_argument("--target_width", type=int, default=1280, help="Evaluation width. Default: 1280.")
    parser.add_argument("--target_height", type=int, default=720, help="Evaluation height. Default: 720.")
    parser.add_argument(
        "--resize_interp",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Interpolation used when resizing GT/pred depth and RGB to the evaluation size.",
    )
    parser.add_argument(
        "--pred_polarity",
        type=str,
        default="auto",
        choices=["auto", "far_white", "near_white"],
        help=(
            "How prediction brightness maps to depth ordering. "
            "'far_white' means white=far, 'near_white' means white=near, "
            "and 'auto' chooses the orientation with higher Spearman rho."
        ),
    )
    parser.add_argument(
        "--delta05_threshold",
        type=float,
        default=math.sqrt(1.25),
        help=(
            "Threshold for δ0.5-AI. Default: sqrt(1.25)≈1.118, i.e. "
            "mean(max(pred/gt, gt/pred) < sqrt(1.25)) after affine alignment."
        ),
    )
    parser.add_argument(
        "--boundary_t_min",
        type=float,
        default=1.05,
        help=(
            "Minimum relative depth-ratio threshold for DepthPro-style scale-invariant "
            "boundary F1. Default: 1.05."
        ),
    )
    parser.add_argument(
        "--boundary_t_max",
        type=float,
        default=1.25,
        help=(
            "Maximum relative depth-ratio threshold for DepthPro-style scale-invariant "
            "boundary F1. Default: 1.25."
        ),
    )
    parser.add_argument(
        "--boundary_num_thresholds",
        type=int,
        default=10,
        help=(
            "Number of linearly spaced thresholds used by DepthPro-style boundary F1. "
            "The per-threshold F1 values are averaged with weights proportional to the thresholds. Default: 10."
        ),
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Threshold in [0,1] for optional GT masks. Default: 0.5.",
    )
    parser.add_argument(
        "--disable_mask",
        action="store_true",
        help="Ignore GT masks even when matching *_mask.png files exist.",
    )
    parser.add_argument(
        "--require_mask",
        action="store_true",
        help="Fail an image if its matching mask is missing.",
    )
    parser.add_argument(
        "--edge_percentile",
        type=float,
        default=90.0,
        help=(
            "Deprecated/ignored. Kept only for backward-compatible CLI calls; "
            "F1 now uses DepthPro-style relative-ratio boundary evaluation."
        ),
    )
    parser.add_argument(
        "--edge_tolerance_px",
        type=int,
        default=2,
        help=(
            "Deprecated/ignored. Kept only for backward-compatible CLI calls; "
            "DepthPro-style boundary F1 does not use dilation tolerance."
        ),
    )
    parser.add_argument(
        "--kendall_max_pixels",
        type=int,
        default=20000,
        help=(
            "Maximum valid pixels sampled per image for Kendall's tau. "
            "0 means use all valid pixels, which can be slow. Default: 20000."
        ),
    )
    parser.add_argument(
        "--kendall_max_pairs",
        type=int,
        default=200000,
        help=(
            "Fallback pair samples for Kendall's tau when scipy is unavailable. "
            "Default: 200000."
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Base random seed for sampled Kendall's tau. Default: 12345.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional cap on the number of matched images to evaluate. 0 means all.",
    )
    return parser.parse_args()


def pil_interp_mode(name: str) -> int:
    name = name.lower()
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unknown interpolation: {name}")


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_gray_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in {"I", "I;16", "I;16B", "I;16L", "F", "L"}:
            gray = image.copy()
        elif image.mode in {"LA", "RGBA"}:
            gray = image.getchannel(0)
        else:
            gray = image.convert("L")
    return np.asarray(gray, dtype=np.float64)


def read_depth_prediction(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
        if values.ndim > 2:
            values = np.squeeze(values)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D depth array in {path}, got shape {values.shape}")
        return np.asarray(values, dtype=np.float64)
    if suffix == ".png":
        return read_gray_png(path)
    raise ValueError(f"Unsupported prediction format: {path}")


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def resize_gray(values: np.ndarray, size: Tuple[int, int], interp: int) -> np.ndarray:
    width, height = size
    image = Image.fromarray(values.astype(np.float32), mode="F")
    resized = image.resize((width, height), interp)
    return np.asarray(resized, dtype=np.float64)


def resize_rgb(values: np.ndarray, size: Tuple[int, int], interp: int) -> np.ndarray:
    width, height = size
    image = Image.fromarray(np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8), mode="RGB")
    resized = image.resize((width, height), interp)
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_mask(values: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    image = Image.fromarray(np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8), mode="L")
    resized = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) / 255.0


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end

    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    return pearson_corr(rankdata(a), rankdata(b))


def normalize_minmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape, dtype=np.float64)
    valid = values[mask]
    if valid.size == 0:
        return output
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value:
        output[mask] = 0.0
    else:
        output[mask] = (values[mask] - min_value) / (max_value - min_value)
    return output


def affine_align(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    if pred.size == 0:
        return float("nan"), float("nan")
    x = np.stack([pred, np.ones_like(pred)], axis=1)
    scale, shift = np.linalg.lstsq(x, gt, rcond=None)[0]
    return float(scale), float(shift)


def unit_scale_to_meter(raw_depth: np.ndarray) -> float:
    valid = raw_depth[np.isfinite(raw_depth) & (raw_depth > 0.0)]
    if valid.size == 0:
        return 1.0
    p95 = float(np.percentile(valid, 95))
    return 0.001 if p95 > 100.0 else 1.0


def load_exr_depth(path: Path) -> np.ndarray:
    if OpenEXR is None or Imath is None:
        raise RuntimeError("OpenEXR and Imath are required to read EXR depth GT files.")

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    channel_names = list(header["channels"].keys())

    channel_name = None
    for candidate in ("R", "Y", "Z"):
        if candidate in channel_names:
            channel_name = candidate
            break
    if channel_name is None:
        channel_name = channel_names[0]

    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    buffer = exr.channel(channel_name, pixel_type)
    depth = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)
    return depth.astype(np.float64)


def load_dat_depth(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"DAT depth file is too small: {path}")

    header = np.frombuffer(raw[:8], dtype="<u4")
    dim_a = int(header[0])
    dim_b = int(header[1])
    payload = np.frombuffer(raw[8:], dtype="<f4")

    if payload.size == dim_a * dim_b:
        height, width = dim_a, dim_b
    elif payload.size == dim_b * dim_a:
        height, width = dim_b, dim_a
    else:
        raise ValueError(
            f"DAT payload size mismatch for {path}: header=({dim_a}, {dim_b}), payload_floats={payload.size}"
        )

    depth = payload.reshape(height, width)
    return depth.astype(np.float64)


def load_gt_depth_in_meters(path: Path) -> Tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".exr":
        raw_depth = load_exr_depth(path)
    elif suffix == ".dat":
        raw_depth = load_dat_depth(path)
    else:
        raise ValueError(f"Unsupported GT raw depth format: {path}")

    scale = unit_scale_to_meter(raw_depth)
    depth_m = raw_depth.astype(np.float64) * scale
    depth_m[~np.isfinite(depth_m)] = np.nan
    depth_m[depth_m <= 0.0] = np.nan
    return depth_m, suffix.lstrip(".")


def strip_known_suffix(name: str, suffixes: Sequence[str]) -> Optional[str]:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def collect_prediction_index(pred_root: Path) -> Dict[Path, Path]:
    index: Dict[Path, Path] = {}
    pred_paths = sorted(path for path in pred_root.rglob("*") if path.is_file())
    for pred_path in pred_paths:
        stem = strip_known_suffix(pred_path.name, PRED_DEPTH_SUFFIXES)
        if stem is None:
            continue
        parent = pred_path.relative_to(pred_root).parent
        if parent.name == "relative_depth":
            parent = parent.parent
        rel_key = parent / stem
        index.setdefault(rel_key, pred_path)
    return index


def parse_include_category_contains(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def category_is_included(category: str, include_category_contains: Sequence[str]) -> bool:
    if not include_category_contains:
        return True
    return any(part in category for part in include_category_contains)


def gt_format_matches(path: Path, gt_format: str) -> bool:
    suffix = path.suffix.lower()
    if gt_format == "auto":
        return suffix in {".dat", ".exr"}
    return suffix == f".{gt_format}"


def gt_preference_rank(path: Path, gt_format: str) -> int:
    suffix = path.suffix.lower()
    if gt_format == "dat":
        return 0 if suffix == ".dat" else 1
    if gt_format == "exr":
        return 0 if suffix == ".exr" else 1
    return 0 if suffix == ".dat" else 1


def collect_gt_index(gt_root: Path, gt_format: str) -> Dict[Path, Path]:
    gt_index: Dict[Path, Path] = {}
    raw_paths = sorted(
        path
        for path in gt_root.rglob("*")
        if (
            path.is_file()
            and any(path.name.endswith(suffix) for suffix in GT_DEPTH_SUFFIXES)
            and gt_format_matches(path, gt_format)
        )
    )

    for raw_path in raw_paths:
        rel = raw_path.relative_to(gt_root)
        frame_id = strip_known_suffix(raw_path.name, GT_DEPTH_SUFFIXES)
        if frame_id is None:
            continue

        gt_key = rel.parent / frame_id
        existing = gt_index.get(gt_key)
        if existing is None or gt_preference_rank(raw_path, gt_format) < gt_preference_rank(existing, gt_format):
            gt_index[gt_key] = raw_path

    return gt_index


def find_mask_path(mask_root: Optional[Path], gt_key: Path, raw_path: Path, frame_id: str) -> Optional[Path]:
    candidates: List[Path] = []
    if mask_root is not None:
        candidates.append(mask_root / gt_key.parent / f"{frame_id}{MASK_SUFFIX}")
        candidates.append(mask_root / f"{frame_id}{MASK_SUFFIX}")
    candidates.append(raw_path.with_name(f"{frame_id}{MASK_SUFFIX}"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def collect_pairs(
    gt_root: Path,
    pred_root: Path,
    max_images: int,
    gt_format: str,
    mask_root: Optional[Path],
    include_category_contains: Sequence[str],
) -> Tuple[List[PairInfo], List[str]]:
    pairs: List[PairInfo] = []
    missing_predictions: List[str] = []
    pred_index = collect_prediction_index(pred_root)
    gt_index = collect_gt_index(gt_root, gt_format)

    for gt_key, raw_path in sorted(gt_index.items()):
        rel = raw_path.relative_to(gt_root)
        if len(rel.parts) >= 3:
            category = rel.parts[0]
            scene = rel.parts[1]
        else:
            category = ""
            scene = rel.parent.as_posix()

        if not category_is_included(category, include_category_contains):
            continue

        frame_id = strip_known_suffix(raw_path.name, GT_DEPTH_SUFFIXES)
        if frame_id is None:
            continue

        rgb_path = raw_path.with_name(f"{frame_id}{RGB_SUFFIX}")
        if not rgb_path.exists():
            rgb_path = None

        mask_path = find_mask_path(
            mask_root=mask_root,
            gt_key=gt_key,
            raw_path=raw_path,
            frame_id=frame_id,
        )

        pred_key = gt_key
        pred_path = pred_index.get(pred_key)
        name = gt_key.as_posix()

        if pred_path is None:
            missing_predictions.append(name)
            continue

        pairs.append(
            PairInfo(
                name=name,
                category=category,
                scene=scene,
                frame_id=frame_id,
                gt_raw_path=raw_path,
                gt_rgb_path=rgb_path,
                gt_mask_path=mask_path,
                pred_path=pred_path,
            )
        )

    if max_images > 0:
        pairs = pairs[:max_images]

    return pairs, missing_predictions


def collect_extra_predictions(
    pred_root: Path,
    gt_root: Path,
    gt_format: str,
    include_category_contains: Sequence[str],
) -> List[str]:
    gt_expected = {
        key.as_posix()
        for key in collect_gt_index(gt_root, gt_format)
        if category_is_included(key.parts[0] if key.parts else "", include_category_contains)
    }

    extras = []
    for pred_key, path in sorted(collect_prediction_index(pred_root).items()):
        if pred_key.as_posix() not in gt_expected:
            extras.append(path.relative_to(pred_root).as_posix())
    return extras


def pseudo_rgb_from_depth(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    gray = normalize_minmax(np.nan_to_num(depth_m, nan=0.0), valid_mask)
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)


def prepare_gt(
    pair: PairInfo,
    target_size: Tuple[int, int],
    resize_interp: int,
    mask_threshold: float,
    use_mask: bool,
    require_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, int, int]:
    depth_m, source_format = load_gt_depth_in_meters(pair.gt_raw_path)

    depth_valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    valid_mask = depth_valid_mask.copy()
    mask_used = False
    mask_pixels = int(depth_valid_mask.sum())
    mask_excluded_pixels = 0
    if use_mask:
        if pair.gt_mask_path is None:
            if require_mask:
                raise FileNotFoundError(f"Missing GT mask for {pair.name}")
        else:
            mask_values = read_mask(pair.gt_mask_path)
            if mask_values.shape != depth_m.shape:
                mask_values = resize_mask(mask_values, (depth_m.shape[1], depth_m.shape[0]))
            file_mask = mask_values >= mask_threshold
            mask_pixels = int((depth_valid_mask & file_mask).sum())
            mask_excluded_pixels = int((depth_valid_mask & ~file_mask).sum())
            valid_mask &= file_mask
            mask_used = True

    if pair.gt_rgb_path is not None:
        rgb = read_rgb(pair.gt_rgb_path)
    else:
        rgb = pseudo_rgb_from_depth(depth_m, valid_mask)

    gt_depth = resize_gray(np.nan_to_num(depth_m, nan=0.0), target_size, resize_interp)
    gt_rgb = resize_rgb(rgb, target_size, resize_interp)
    gt_mask = resize_mask(valid_mask.astype(np.float32), target_size) >= mask_threshold
    gt_mask &= np.isfinite(gt_depth) & (gt_depth > 0.0)
    return gt_depth, gt_rgb, gt_mask.astype(bool), source_format, mask_used, mask_pixels, mask_excluded_pixels


def prepare_prediction(pred_path: Path, target_size: Tuple[int, int], resize_interp: int) -> Tuple[np.ndarray, int, int]:
    pred = read_depth_prediction(pred_path)
    pred_height, pred_width = pred.shape
    if (pred_width, pred_height) != target_size:
        pred = resize_gray(pred, target_size, resize_interp)
    return pred, pred_width, pred_height


def choose_prediction_orientation(
    gt_depth: np.ndarray,
    pred_raw: np.ndarray,
    valid_mask: np.ndarray,
    pred_polarity: str,
) -> Tuple[np.ndarray, str, float]:
    pred_norm = normalize_minmax(pred_raw, valid_mask)
    candidates: List[Tuple[str, np.ndarray]] = []

    if pred_polarity in {"auto", "far_white"}:
        candidates.append(("far_white", pred_norm))
    if pred_polarity in {"auto", "near_white"}:
        candidates.append(("near_white", 1.0 - pred_norm))

    if not candidates:
        raise ValueError(f"Invalid pred_polarity: {pred_polarity}")

    gt_valid = gt_depth[valid_mask]
    best_name = candidates[0][0]
    best_pred = candidates[0][1]
    best_score = spearman_corr(gt_valid, best_pred[valid_mask])

    for name, candidate in candidates[1:]:
        score = spearman_corr(gt_valid, candidate[valid_mask])
        if not np.isfinite(best_score) or (np.isfinite(score) and score > best_score):
            best_name = name
            best_pred = candidate
            best_score = score

    return best_pred, best_name, float(best_score)


def compute_affine_metrics(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    delta05_threshold: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, np.ndarray]:
    gt_valid = gt_depth[valid_mask]
    pred_valid = pred_depth[valid_mask]
    scale, shift = affine_align(pred_valid, gt_valid)
    aligned_full = scale * pred_depth + shift
    aligned = aligned_full[valid_mask]

    eps = 1e-6
    gt_pos = np.maximum(gt_valid, eps)
    aligned_pos = np.maximum(aligned, eps)

    abs_error = np.abs(aligned - gt_valid)
    squared_error = (aligned - gt_valid) ** 2

    absrel_ai = float(np.mean(abs_error / gt_pos))
    rmse_ai = float(np.sqrt(np.mean(squared_error)))
    rms_ai = rmse_ai
    mae_ai = float(np.mean(abs_error))

    ratio = np.maximum(aligned_pos / gt_pos, gt_pos / aligned_pos)
    delta1_ai = float(np.mean(ratio < 1.25))
    delta2_ai = float(np.mean(ratio < 1.25**2))
    delta3_ai = float(np.mean(ratio < 1.25**3))
    delta05_ai = float(np.mean(ratio < delta05_threshold))
    log10_ai = float(np.mean(np.abs(np.log10(aligned_pos) - np.log10(gt_pos))))
    return (
        scale,
        shift,
        absrel_ai,
        rmse_ai,
        rms_ai,
        mae_ai,
        delta1_ai,
        delta2_ai,
        delta3_ai,
        delta05_ai,
        log10_ai,
        aligned_full,
    )

def depthpro_fgbg_depth(
    depth: np.ndarray,
    threshold: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DepthPro foreground/background relations between neighboring pixels.

    Returns directional depth-ratio contours for left, top, right, and bottom
    neighbor pairs. The array shapes follow DepthPro's official code:
    left/right are H x (W-1), and top/bottom are (H-1) x W.
    """
    eps = 1e-6
    depth = np.asarray(depth, dtype=np.float64)
    depth = np.maximum(depth, eps)

    right_is_big_enough = (depth[:, 1:] / depth[:, :-1]) > threshold
    left_is_big_enough = (depth[:, :-1] / depth[:, 1:]) > threshold
    bottom_is_big_enough = (depth[1:, :] / depth[:-1, :]) > threshold
    top_is_big_enough = (depth[:-1, :] / depth[1:, :]) > threshold

    if valid_mask is not None:
        valid_mask = valid_mask.astype(bool)
        horizontal_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        vertical_valid = valid_mask[1:, :] & valid_mask[:-1, :]
        right_is_big_enough &= horizontal_valid
        left_is_big_enough &= horizontal_valid
        bottom_is_big_enough &= vertical_valid
        top_is_big_enough &= vertical_valid

    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def depthpro_boundary_f1(
    predicted_inverse_depth: np.ndarray,
    target_inverse_depth: np.ndarray,
    threshold: float,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """DepthPro boundary F1 for a single relative depth-ratio threshold."""
    ap, bp, cp, dp = depthpro_fgbg_depth(predicted_inverse_depth, threshold, valid_mask)
    ag, bg, cg, dg = depthpro_fgbg_depth(target_inverse_depth, threshold, valid_mask)

    recall = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )

    precision = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ap), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bp), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cp), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dp), 1)
    )

    if recall + precision == 0.0:
        return 0.0
    return float(2.0 * recall * precision / (recall + precision))


def depthpro_thresholds_and_weights(t_min: float, t_max: float, count: int) -> Tuple[np.ndarray, np.ndarray]:
    thresholds = np.linspace(float(t_min), float(t_max), int(count), dtype=np.float64)
    weights = thresholds / thresholds.sum()
    return thresholds, weights


def invert_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return 1.0 / np.asarray(depth, dtype=np.float64).clip(min=eps)


def compute_edge_detail_f1(
    gt_depth: np.ndarray,
    aligned_pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    boundary_t_min: float,
    boundary_t_max: float,
    boundary_num_thresholds: int,
) -> float:
    """DepthPro-style scale-invariant boundary F1.

    DepthPro computes occluding-contour relations from pairwise neighboring-pixel
    depth ratios, evaluates F1 over thresholds linearly spaced from t_min to t_max,
    and averages per-threshold F1 scores with weights proportional to thresholds.
    The official implementation applies this to inverse depth; we do the same.
    """
    valid = (
        valid_mask.astype(bool)
        & np.isfinite(gt_depth)
        & np.isfinite(aligned_pred_depth)
        & (gt_depth > 0.0)
    )
    if int(valid.sum()) < 2:
        return float("nan")

    thresholds, weights = depthpro_thresholds_and_weights(
        t_min=boundary_t_min,
        t_max=boundary_t_max,
        count=boundary_num_thresholds,
    )
    pred_inv = invert_depth(aligned_pred_depth)
    gt_inv = invert_depth(gt_depth)
    f1_scores = np.asarray(
        [
            depthpro_boundary_f1(
                predicted_inverse_depth=pred_inv,
                target_inverse_depth=gt_inv,
                threshold=float(threshold),
                valid_mask=valid,
            )
            for threshold in thresholds
        ],
        dtype=np.float64,
    )
    return float(np.sum(f1_scores * weights))

def make_rng_for_pair(base_seed: int, name: str) -> np.random.Generator:
    name_hash = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    seed = (int(base_seed) + name_hash) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def approximate_kendall_from_pairs(
    gt_values: np.ndarray,
    pred_values: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> float:
    n = gt_values.size
    if n < 2 or max_pairs <= 0:
        return float("nan")

    pair_count = int(max_pairs)
    idx_a = rng.integers(0, n, size=pair_count, endpoint=False)
    idx_b = rng.integers(0, n, size=pair_count, endpoint=False)
    keep = idx_a != idx_b
    if not np.any(keep):
        return float("nan")

    idx_a = idx_a[keep]
    idx_b = idx_b[keep]

    gt_sign = np.sign(gt_values[idx_a] - gt_values[idx_b])
    pred_sign = np.sign(pred_values[idx_a] - pred_values[idx_b])
    informative = (gt_sign != 0.0) & (pred_sign != 0.0)

    if not np.any(informative):
        return float("nan")

    return float(np.mean(gt_sign[informative] * pred_sign[informative]))


def compute_kendall_tau(
    gt_values: np.ndarray,
    pred_values: np.ndarray,
    max_pixels: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> float:
    finite = np.isfinite(gt_values) & np.isfinite(pred_values)
    gt_values = gt_values[finite]
    pred_values = pred_values[finite]

    if gt_values.size < 2:
        return float("nan")

    if max_pixels > 0 and gt_values.size > max_pixels:
        sample_ids = rng.choice(gt_values.size, size=max_pixels, replace=False)
        gt_values = gt_values[sample_ids]
        pred_values = pred_values[sample_ids]

    if scipy_kendalltau is not None:
        result = scipy_kendalltau(gt_values, pred_values, nan_policy="omit")
        tau = result.statistic if hasattr(result, "statistic") else result[0]
        return float(tau) if np.isfinite(tau) else float("nan")

    return approximate_kendall_from_pairs(
        gt_values=gt_values,
        pred_values=pred_values,
        max_pairs=max_pairs,
        rng=rng,
    )

def ordinal_accuracy_from_pairs(
    gt_values: np.ndarray,
    pred_values: np.ndarray,
    pairs: np.ndarray,
    epsilon: float,
) -> float:
    if pairs.size == 0:
        return float("nan")

    gt_diff = gt_values[pairs[:, 0]] - gt_values[pairs[:, 1]]
    informative = np.abs(gt_diff) > epsilon
    if not np.any(informative):
        return float("nan")

    pred_diff = pred_values[pairs[:, 0]] - pred_values[pairs[:, 1]]
    correct = np.sign(gt_diff[informative]) == np.sign(pred_diff[informative])
    return float(correct.mean())


def evaluate_pair(pair: PairInfo, args: argparse.Namespace) -> ImageMetrics:
    target_size = (args.target_width, args.target_height)
    resize_interp = pil_interp_mode(args.resize_interp)

    gt_depth, gt_rgb, valid_mask, source_format, mask_used, mask_pixels, mask_excluded_pixels = prepare_gt(
        pair=pair,
        target_size=target_size,
        resize_interp=resize_interp,
        mask_threshold=args.mask_threshold,
        use_mask=not args.disable_mask,
        require_mask=args.require_mask,
    )
    pred_raw, pred_width, pred_height = prepare_prediction(
        pair.pred_path,
        target_size=target_size,
        resize_interp=resize_interp,
    )

    valid_mask &= np.isfinite(pred_raw)
    valid_pixels = int(valid_mask.sum())
    if valid_pixels < 2:
        raise ValueError("Not enough valid pixels after masking.")

    pred_depth, chosen_polarity, spearman_rho = choose_prediction_orientation(
        gt_depth=gt_depth,
        pred_raw=pred_raw,
        valid_mask=valid_mask,
        pred_polarity=args.pred_polarity,
    )

    (
        affine_scale,
        affine_shift,
        absrel_ai,
        rmse_ai,
        rms_ai,
        mae_ai,
        delta1_ai,
        delta2_ai,
        delta3_ai,
        delta05_ai,
        log10_ai,
        aligned_pred_depth,
    ) = compute_affine_metrics(
        gt_depth=gt_depth,
        pred_depth=pred_depth,
        valid_mask=valid_mask,
        delta05_threshold=args.delta05_threshold,
    )

    edge_detail_f1 = compute_edge_detail_f1(
        gt_depth=gt_depth,
        aligned_pred_depth=aligned_pred_depth,
        valid_mask=valid_mask,
        boundary_t_min=args.boundary_t_min,
        boundary_t_max=args.boundary_t_max,
        boundary_num_thresholds=args.boundary_num_thresholds,
    )

    kendall_tau = compute_kendall_tau(
        gt_values=gt_depth[valid_mask],
        pred_values=pred_depth[valid_mask],
        max_pixels=args.kendall_max_pixels,
        max_pairs=args.kendall_max_pairs,
        rng=make_rng_for_pair(args.random_seed, pair.name),
    )

    return ImageMetrics(
        name=pair.name,
        category=pair.category,
        scene=pair.scene,
        frame_id=pair.frame_id,
        gt_raw_path=str(pair.gt_raw_path),
        gt_rgb_path=str(pair.gt_rgb_path) if pair.gt_rgb_path is not None else "",
        gt_mask_path=str(pair.gt_mask_path) if pair.gt_mask_path is not None else "",
        pred_path=str(pair.pred_path),
        source_format=source_format,
        width=args.target_width,
        height=args.target_height,
        pred_width=pred_width,
        pred_height=pred_height,
        resized_pred=(pred_width, pred_height) != target_size,
        resize_method=args.resize_interp,
        pred_polarity=chosen_polarity,
        valid_pixels=valid_pixels,
        mask_used=mask_used,
        mask_pixels=mask_pixels,
        mask_excluded_pixels=mask_excluded_pixels,
        affine_scale=affine_scale,
        affine_shift=affine_shift,
        spearman_rho=spearman_rho,
        kendall_tau=kendall_tau,
        absrel_ai=absrel_ai,
        rmse_ai=rmse_ai,
        rms_ai=rms_ai,
        mae_ai=mae_ai,
        delta1_ai=delta1_ai,
        delta2_ai=delta2_ai,
        delta3_ai=delta3_ai,
        delta05_ai=delta05_ai,
        log10_ai=log10_ai,
        edge_detail_f1=edge_detail_f1,
    )


def mean_valid(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def summarize_rows(rows: Sequence[ImageMetrics]) -> Dict[str, float]:
    return {
        "count": len(rows),
        "valid_pixels_mean": mean_valid([row.valid_pixels for row in rows]),
        "masks_used_count": sum(1 for row in rows if row.mask_used),
        "mask_excluded_pixels_mean": mean_valid([row.mask_excluded_pixels for row in rows]),
        "spearman_rho_mean": mean_valid([row.spearman_rho for row in rows]),
        "kendall_tau_mean": mean_valid([row.kendall_tau for row in rows]),
        "absrel_ai_mean": mean_valid([row.absrel_ai for row in rows]),
        "rmse_ai_mean": mean_valid([row.rmse_ai for row in rows]),
        "rms_ai_mean": mean_valid([row.rms_ai for row in rows]),
        "mae_ai_mean": mean_valid([row.mae_ai for row in rows]),
        "delta1_ai_mean": mean_valid([row.delta1_ai for row in rows]),
        "delta2_ai_mean": mean_valid([row.delta2_ai for row in rows]),
        "delta3_ai_mean": mean_valid([row.delta3_ai for row in rows]),
        "delta05_ai_mean": mean_valid([row.delta05_ai for row in rows]),
        "log10_ai_mean": mean_valid([row.log10_ai for row in rows]),
        "edge_detail_f1_mean": mean_valid([row.edge_detail_f1 for row in rows]),
        "affine_scale_mean": mean_valid([row.affine_scale for row in rows]),
        "affine_shift_mean": mean_valid([row.affine_shift for row in rows]),
    }


def rows_to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: rows_to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [rows_to_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [rows_to_json_safe(value) for value in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def save_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_results(
    output_dir: Path,
    metrics: Sequence[ImageMetrics],
    overall_summary: Dict[str, Any],
    category_summaries: Sequence[Dict[str, Any]],
    scene_summaries: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = [asdict(row) for row in metrics]
    save_csv(
        output_dir / "depth_metrics.csv",
        detail_rows,
        fieldnames=list(ImageMetrics.__dataclass_fields__.keys()),
    )

    save_csv(
        output_dir / "depth_category_summary.csv",
        category_summaries,
        fieldnames=list(category_summaries[0].keys()) if category_summaries else ["category", "count"],
    )
    save_csv(
        output_dir / "depth_scene_summary.csv",
        scene_summaries,
        fieldnames=list(scene_summaries[0].keys()) if scene_summaries else ["scene", "count"],
    )

    summary = {
        "overall": overall_summary,
        "by_category": category_summaries,
        "by_scene": scene_summaries,
    }
    with open(output_dir / "depth_summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows_to_json_safe(summary), handle, indent=2, ensure_ascii=False)

    with open(output_dir / "depth_overall_summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows_to_json_safe(overall_summary), handle, indent=2, ensure_ascii=False)


def printable_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "count",
        "evaluated_images",
        "failed_images",
        "valid_pixels_mean",
        "masks_used_count",
        "mask_excluded_pixels_mean",
        "spearman_rho_mean",
        "kendall_tau_mean",
        "absrel_ai_mean",
        "rmse_ai_mean",
        "rms_ai_mean",
        "mae_ai_mean",
        "delta1_ai_mean",
        "delta2_ai_mean",
        "delta3_ai_mean",
        "delta05_ai_mean",
        "log10_ai_mean",
        "edge_detail_f1_mean",
        "affine_scale_mean",
        "affine_shift_mean",
        "missing_prediction_count",
        "extra_prediction_count",
    ]
    return {key: summary[key] for key in keys if key in summary}


def main() -> None:
    args = parse_args()

    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("Target size must be positive.")
    if args.delta05_threshold <= 1.0:
        raise ValueError("--delta05_threshold must be greater than 1.0.")
    if args.boundary_t_min <= 1.0:
        raise ValueError("--boundary_t_min must be greater than 1.0.")
    if args.boundary_t_max <= args.boundary_t_min:
        raise ValueError("--boundary_t_max must be greater than --boundary_t_min.")
    if args.boundary_num_thresholds <= 0:
        raise ValueError("--boundary_num_thresholds must be positive.")
    if args.kendall_max_pixels < 0:
        raise ValueError("--kendall_max_pixels must be non-negative.")
    if args.kendall_max_pairs <= 0:
        raise ValueError("--kendall_max_pairs must be positive.")
    if not (0.0 <= args.mask_threshold <= 1.0):
        raise ValueError("--mask_threshold must be in [0, 1].")

    gt_root = args.gt_root.resolve()
    pred_root = args.pred_root.resolve()
    mask_root = args.mask_root.resolve() if args.mask_root is not None else None
    output_dir = args.output_dir.resolve()
    include_category_contains = parse_include_category_contains(args.include_category_contains)

    pairs, missing_predictions = collect_pairs(
        gt_root=gt_root,
        pred_root=pred_root,
        max_images=args.max_images,
        gt_format=args.gt_format,
        mask_root=mask_root,
        include_category_contains=include_category_contains,
    )
    if not pairs:
        raise RuntimeError("No matched GT/prediction pairs were found.")

    extra_predictions = collect_extra_predictions(
        pred_root=pred_root,
        gt_root=gt_root,
        gt_format=args.gt_format,
        include_category_contains=include_category_contains,
    )

    metrics: List[ImageMetrics] = []
    errors: List[str] = []
    for index, pair in enumerate(pairs, start=1):
        try:
            row = evaluate_pair(pair, args)
            metrics.append(row)
            print(
                f"[{row.name}] "
                f"rho={row.spearman_rho:.6f} | tau={row.kendall_tau:.6f} | "
                f"AbsRel-AI={row.absrel_ai:.6f} | RMSE-AI={row.rmse_ai:.6f} | "
                f"RMS-AI={row.rms_ai:.6f} | MAE-AI={row.mae_ai:.6f} | "
                f"delta1-AI={row.delta1_ai:.6f} | delta2-AI={row.delta2_ai:.6f} | "
                f"delta3-AI={row.delta3_ai:.6f} | delta0.5-AI={row.delta05_ai:.6f} | "
                f"log10-AI={row.log10_ai:.6f} | DepthProF1={row.edge_detail_f1:.6f} | "
                f"polarity={row.pred_polarity} | mask_used={row.mask_used}"
            )
        except Exception as exc:
            message = f"{pair.name}: {exc}"
            errors.append(message)
            print(f"[{pair.name}] ERROR: {exc}")

    if not metrics:
        raise RuntimeError("All image evaluations failed.")

    overall_summary = summarize_rows(metrics)
    overall_summary.update(
        {
            "gt_root": str(gt_root),
            "pred_root": str(pred_root),
            "mask_root": str(mask_root) if mask_root is not None else "",
            "output_dir": str(output_dir),
            "gt_format": args.gt_format,
            "include_category_contains": list(include_category_contains),
            "target_width": args.target_width,
            "target_height": args.target_height,
            "resize_interp": args.resize_interp,
            "pred_polarity_setting": args.pred_polarity,
            "mask_enabled": not args.disable_mask,
            "mask_required": args.require_mask,
            "mask_threshold": args.mask_threshold,
            "delta05_threshold": args.delta05_threshold,
            "boundary_t_min": args.boundary_t_min,
            "boundary_t_max": args.boundary_t_max,
            "boundary_num_thresholds": args.boundary_num_thresholds,
            "edge_percentile_deprecated": args.edge_percentile,
            "edge_tolerance_px_deprecated": args.edge_tolerance_px,
            "kendall_max_pixels": args.kendall_max_pixels,
            "kendall_max_pairs": args.kendall_max_pairs,
            "random_seed": args.random_seed,
            "evaluated_images": len(metrics),
            "failed_images": len(errors),
            "missing_prediction_count": len(missing_predictions),
            "extra_prediction_count": len(extra_predictions),
            "missing_predictions": missing_predictions,
            "extra_predictions": extra_predictions,
            "errors": errors,
        }
    )

    category_summaries: List[Dict[str, Any]] = []
    for category in sorted({row.category for row in metrics}):
        rows = [row for row in metrics if row.category == category]
        summary = summarize_rows(rows)
        summary["category"] = category
        category_summaries.append({"category": category, **summary})

    scene_summaries: List[Dict[str, Any]] = []
    for scene_key in sorted({f"{row.category}/{row.scene}" for row in metrics}):
        category, scene = scene_key.split("/", 1)
        rows = [row for row in metrics if row.category == category and row.scene == scene]
        summary = summarize_rows(rows)
        scene_summaries.append({"category": category, "scene": scene, **summary})

    save_results(
        output_dir=output_dir,
        metrics=metrics,
        overall_summary=overall_summary,
        category_summaries=category_summaries,
        scene_summaries=scene_summaries,
    )

    print("\n=== Summary ===")
    for key, value in printable_summary(overall_summary).items():
        print(f"{key}: {value}")
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
