#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate predicted normal maps against GT normal maps.

Metrics:
- Mean Angular Error (degree)
- Median Angular Error (degree)
- RMSE Angular Error (degree)
- Acc@5 / Acc@7.5 / Acc@11.25 / Acc@22.5 / Acc@30

Predictions are resized to GT resolution before evaluation.
Normal RGB is decoded by default as n = 2 * RGB - 1, then L2-normalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class NormalMetrics:
    name: str
    width: int
    height: int
    mean_angular: float
    median_angular: float
    rmse_angular: float
    acc_5: float
    acc_7_5: float
    acc_11_25: float
    acc_22_5: float
    acc_30: float
    valid_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted normal maps against GT normal maps."
    )

    parser.add_argument("--gt_path", type=str, default="", help="Single GT normal path.")
    parser.add_argument("--pred_path", type=str, default="", help="Single predicted normal path.")
    parser.add_argument("--mask_path", type=str, default="", help="Optional single valid-mask path.")

    parser.add_argument("--gt_dir", type=str, default="", help="GT normal directory.")
    parser.add_argument("--pred_dir", type=str, default="", help="Prediction directory.")
    parser.add_argument("--mask_dir", type=str, default="", help="Optional valid-mask directory.")

    parser.add_argument("--pair_mode", type=str, default="stem", choices=["stem", "name"])
    parser.add_argument("--gt_suffix_to_strip", type=str, default="")
    parser.add_argument("--pred_suffix_to_strip", type=str, default="")
    parser.add_argument("--mask_suffix_to_strip", type=str, default="")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_resized_preds", action="store_true")

    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)

    parser.add_argument(
        "--resize_interp",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
    )
    parser.add_argument("--mask_threshold", type=float, default=0.5)

    parser.add_argument(
        "--normal_encoding",
        type=str,
        default="rgb01",
        choices=["rgb01"],
        help="Decode RGB [0,1] as normal = 2*RGB - 1.",
    )
    parser.add_argument("--gt_flip_x", action="store_true")
    parser.add_argument("--gt_flip_y", action="store_true")
    parser.add_argument("--gt_flip_z", action="store_true")
    # Prediction alignment.
    # For Lotus / StableNormal-style predictions in your current setup,
    # calibration shows that the prediction X axis is opposite to GT:
    #     pred' = (-x, y, z)
    #
    # Therefore pred_flip_x defaults to True.
    # pred_flip_y and pred_flip_z default to False.
    parser.add_argument("--pred_flip_x", dest="pred_flip_x", action="store_true")
    parser.add_argument("--no_pred_flip_x", dest="pred_flip_x", action="store_false")

    parser.add_argument("--pred_flip_y", dest="pred_flip_y", action="store_true")
    parser.add_argument("--no_pred_flip_y", dest="pred_flip_y", action="store_false")
    parser.set_defaults(pred_flip_y=False)

    parser.add_argument("--pred_flip_z", dest="pred_flip_z", action="store_true")
    parser.add_argument("--no_pred_flip_z", dest="pred_flip_z", action="store_false")
    parser.set_defaults(pred_flip_z=False)
    
    parser.add_argument("--min_norm", type=float, default=1e-6)
    parser.add_argument("--ignore_back_facing_gt", action="store_true")

    return parser.parse_args()


def pil_interp_mode(name: str) -> int:
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unknown interpolation: {name}")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def image_to_float_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path, size: Tuple[int, int], threshold: float) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != size:
            img = img.resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr >= threshold


def decode_normal_rgb(
    img_np: np.ndarray,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
    min_norm: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    n = img_np * 2.0 - 1.0

    if flip_x:
        n[..., 0] *= -1.0
    if flip_y:
        n[..., 1] *= -1.0
    if flip_z:
        n[..., 2] *= -1.0

    finite = np.isfinite(n).all(axis=2)
    norm = np.linalg.norm(n, axis=2)
    legal = finite & (norm > min_norm)

    n_unit = np.zeros_like(n, dtype=np.float32)
    n_unit[legal] = n[legal] / norm[legal, None]
    return n_unit, legal


def angular_values_deg(gt_n: np.ndarray, pred_n: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.sum() == 0:
        return np.asarray([], dtype=np.float32)
    dot = np.sum(gt_n * pred_n, axis=2)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.degrees(np.arccos(dot)).astype(np.float32)
    return ang[valid_mask]


def compute_normal_metrics(
    gt_n: np.ndarray,
    pred_n: np.ndarray,
    valid_mask: np.ndarray,
    name: str,
    width: int,
    height: int,
) -> Tuple[NormalMetrics, float, float, int, Dict[str, int]]:
    angles = angular_values_deg(gt_n, pred_n, valid_mask)
    valid_pixels = int(angles.size)

    if valid_pixels == 0:
        nan = float("nan")
        return (
            NormalMetrics(name, width, height, nan, nan, nan, nan, nan, nan, nan, nan, 0),
            0.0,
            0.0,
            0,
            {"acc_5": 0, "acc_7_5": 0, "acc_11_25": 0, "acc_22_5": 0, "acc_30": 0},
        )

    counts = {
        "acc_5": int(np.sum(angles < 5.0)),
        "acc_7_5": int(np.sum(angles < 7.5)),
        "acc_11_25": int(np.sum(angles < 11.25)),
        "acc_22_5": int(np.sum(angles < 22.5)),
        "acc_30": int(np.sum(angles < 30.0)),
    }

    metrics = NormalMetrics(
        name=name,
        width=width,
        height=height,
        mean_angular=float(np.mean(angles)),
        median_angular=float(np.median(angles)),
        rmse_angular=float(np.sqrt(np.mean(angles ** 2))),
        acc_5=float(counts["acc_5"] / valid_pixels),
        acc_7_5=float(counts["acc_7_5"] / valid_pixels),
        acc_11_25=float(counts["acc_11_25"] / valid_pixels),
        acc_22_5=float(counts["acc_22_5"] / valid_pixels),
        acc_30=float(counts["acc_30"] / valid_pixels),
        valid_pixels=valid_pixels,
    )
    return metrics, float(np.sum(angles)), float(np.sum(angles ** 2)), valid_pixels, counts


def normalize_match_key(name: str, suffix_to_strip: str) -> str:
    stem = Path(name).stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def list_images(directory: Path, recursive: bool) -> List[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS])


def build_pairs_from_dirs(
    gt_dir: Path,
    pred_dir: Path,
    mask_dir: Optional[Path],
    pair_mode: str,
    gt_suffix_to_strip: str,
    pred_suffix_to_strip: str,
    mask_suffix_to_strip: str,
    recursive: bool,
) -> List[Tuple[Path, Path, Optional[Path]]]:
    gt_files = list_images(gt_dir, recursive)
    pred_files = list_images(pred_dir, recursive)
    mask_files = list_images(mask_dir, recursive) if mask_dir is not None else []

    def make_key(path: Path, root: Path, suffix_to_strip: str) -> str:
        rel = path.relative_to(root)
        if pair_mode == "name":
            return rel.as_posix()
        stem = normalize_match_key(path.name, suffix_to_strip)
        if rel.parent == Path("."):
            return stem
        return (rel.parent / stem).as_posix()

    gt_map = {make_key(p, gt_dir, gt_suffix_to_strip): p for p in gt_files}
    pred_map = {make_key(p, pred_dir, pred_suffix_to_strip): p for p in pred_files}
    mask_map = {make_key(p, mask_dir, mask_suffix_to_strip): p for p in mask_files} if mask_files else {}

    common_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    return [(gt_map[k], pred_map[k], mask_map.get(k) if mask_map else None) for k in common_keys]


def evaluate_pair(
    gt_path: Path,
    pred_path: Path,
    mask_path: Optional[Path],
    args: argparse.Namespace,
    resized_pred_dir: Optional[Path],
    name: str,
) -> Tuple[NormalMetrics, float, float, int, Dict[str, int]]:
    gt_img = load_rgb(gt_path)
    pred_img = load_rgb(pred_path)

    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, pil_interp_mode(args.resize_interp))

    if resized_pred_dir is not None:
        resized_pred_dir.mkdir(parents=True, exist_ok=True)
        pred_img.save(resized_pred_dir / pred_path.name)

    gt_np = image_to_float_np(gt_img)
    pred_np = image_to_float_np(pred_img)

    gt_n, gt_legal = decode_normal_rgb(
        gt_np,
        flip_x=args.gt_flip_x,
        flip_y=args.gt_flip_y,
        flip_z=args.gt_flip_z,
        min_norm=args.min_norm,
    )
    pred_n, pred_legal = decode_normal_rgb(
        pred_np,
        flip_x=args.pred_flip_x,
        flip_y=args.pred_flip_y,
        flip_z=args.pred_flip_z,
        min_norm=args.min_norm,
    )

    if mask_path is not None and mask_path.exists():
        mask = load_mask(mask_path, gt_img.size, args.mask_threshold)
    else:
        mask = np.ones((gt_np.shape[0], gt_np.shape[1]), dtype=bool)

    valid = mask & gt_legal & pred_legal
    if args.ignore_back_facing_gt:
        valid &= gt_n[..., 2] > 0

    return compute_normal_metrics(
        gt_n=gt_n,
        pred_n=pred_n,
        valid_mask=valid,
        name=name,
        width=gt_img.size[0],
        height=gt_img.size[1],
    )


def summarize(
    metrics: Sequence[NormalMetrics],
    total_sum_ang: float,
    total_sum_sq_ang: float,
    total_valid: int,
    total_counts: Dict[str, int],
) -> Dict[str, float]:
    def mean_valid(values: List[float]) -> float:
        vals = [float(v) for v in values if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def median_valid(values: List[float]) -> float:
        vals = [float(v) for v in values if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def std_valid(values: List[float]) -> float:
        vals = [float(v) for v in values if np.isfinite(v)]
        return float(np.std(vals)) if vals else float("nan")

    if total_valid > 0:
        mean_micro = total_sum_ang / total_valid
        rmse_micro = math.sqrt(total_sum_sq_ang / total_valid)
        acc_5_micro = total_counts["acc_5"] / total_valid
        acc_7_5_micro = total_counts["acc_7_5"] / total_valid
        acc_11_25_micro = total_counts["acc_11_25"] / total_valid
        acc_22_5_micro = total_counts["acc_22_5"] / total_valid
        acc_30_micro = total_counts["acc_30"] / total_valid
    else:
        mean_micro = rmse_micro = float("nan")
        acc_5_micro = acc_7_5_micro = acc_11_25_micro = acc_22_5_micro = acc_30_micro = float("nan")

    return {
        "count": len(metrics),
        "valid_pixels_total": int(total_valid),

        "mean_angular_mean": mean_valid([m.mean_angular for m in metrics]),
        "median_angular_mean": mean_valid([m.median_angular for m in metrics]),
        "rmse_angular_mean": mean_valid([m.rmse_angular for m in metrics]),
        "acc_5_mean": mean_valid([m.acc_5 for m in metrics]),
        "acc_7_5_mean": mean_valid([m.acc_7_5 for m in metrics]),
        "acc_11_25_mean": mean_valid([m.acc_11_25 for m in metrics]),
        "acc_22_5_mean": mean_valid([m.acc_22_5 for m in metrics]),
        "acc_30_mean": mean_valid([m.acc_30 for m in metrics]),

        "mean_angular_median": median_valid([m.mean_angular for m in metrics]),
        "mean_angular_std": std_valid([m.mean_angular for m in metrics]),

        "mean_angular_micro": float(mean_micro),
        "rmse_angular_micro": float(rmse_micro),
        "acc_5_micro": float(acc_5_micro),
        "acc_7_5_micro": float(acc_7_5_micro),
        "acc_11_25_micro": float(acc_11_25_micro),
        "acc_22_5_micro": float(acc_22_5_micro),
        "acc_30_micro": float(acc_30_micro),
    }


def save_results(output_dir: Path, metrics: Sequence[NormalMetrics], summary: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "normal_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "width", "height",
                "mean_angular", "median_angular", "rmse_angular",
                "acc_5", "acc_7_5", "acc_11_25", "acc_22_5", "acc_30",
                "valid_pixels",
            ],
        )
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))

    with open(output_dir / "normal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    single_mode = bool(args.gt_path and args.pred_path)
    dir_mode = bool(args.gt_dir and args.pred_dir)
    if single_mode == dir_mode:
        raise ValueError("Use either single-pair mode (--gt_path/--pred_path) or directory mode (--gt_dir/--pred_dir).")

    output_dir = Path(args.output_dir)
    resized_pred_dir = output_dir / "resized_predictions" if args.save_resized_preds else None

    if single_mode:
        pairs = [(Path(args.gt_path), Path(args.pred_path), Path(args.mask_path) if args.mask_path else None)]
        gt_dir = None
    else:
        gt_dir = Path(args.gt_dir)
        pred_dir = Path(args.pred_dir)
        mask_dir = Path(args.mask_dir) if args.mask_dir else None
        pairs = build_pairs_from_dirs(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            mask_dir=mask_dir,
            pair_mode=args.pair_mode,
            gt_suffix_to_strip=args.gt_suffix_to_strip,
            pred_suffix_to_strip=args.pred_suffix_to_strip,
            mask_suffix_to_strip=args.mask_suffix_to_strip,
            recursive=args.recursive,
        )
        if len(pairs) == 0:
            raise RuntimeError("No matched GT/prediction pairs found.")

    metrics: List[NormalMetrics] = []
    total_sum_ang = 0.0
    total_sum_sq_ang = 0.0
    total_valid = 0
    total_counts = {"acc_5": 0, "acc_7_5": 0, "acc_11_25": 0, "acc_22_5": 0, "acc_30": 0}

    for gt_path, pred_path, mask_path in pairs:
        name = gt_path.stem if single_mode else gt_path.relative_to(gt_dir).as_posix()
        m, sum_ang, sum_sq_ang, valid_pixels, counts = evaluate_pair(
            gt_path=gt_path,
            pred_path=pred_path,
            mask_path=mask_path,
            args=args,
            resized_pred_dir=resized_pred_dir,
            name=name,
        )
        metrics.append(m)
        total_sum_ang += sum_ang
        total_sum_sq_ang += sum_sq_ang
        total_valid += valid_pixels
        for k in total_counts:
            total_counts[k] += counts[k]

        print(
            f"[{m.name}] mean={m.mean_angular:.4f} | median={m.median_angular:.4f} | "
            f"rmse={m.rmse_angular:.4f} | acc11.25={m.acc_11_25:.4f} | "
            f"acc22.5={m.acc_22_5:.4f} | acc30={m.acc_30:.4f} | valid={m.valid_pixels}"
        )

    summary = summarize(
        metrics=metrics,
        total_sum_ang=total_sum_ang,
        total_sum_sq_ang=total_sum_sq_ang,
        total_valid=total_valid,
        total_counts=total_counts,
    )
    save_results(output_dir=output_dir, metrics=metrics, summary=summary)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
