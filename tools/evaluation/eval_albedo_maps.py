#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Optional dependencies
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    import torch
except Exception:
    torch = None

try:
    import lpips  # pip install lpips
except Exception:
    lpips = None


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageMetrics:
    name: str
    width: int
    height: int
    mae: float
    psnr: float
    ssim: float
    lpips: Optional[float]
    valid_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted albedo maps against GT after resizing predictions to GT resolution."
    )

    # Single-pair mode
    parser.add_argument("--gt_path", type=str, default="", help="Single GT image path.")
    parser.add_argument("--pred_path", type=str, default="", help="Single predicted image path.")
    parser.add_argument("--mask_path", type=str, default="", help="Optional single valid-mask path.")

    # Directory mode
    parser.add_argument("--gt_dir", type=str, default="", help="GT directory.")
    parser.add_argument("--pred_dir", type=str, default="", help="Prediction directory.")
    parser.add_argument("--mask_dir", type=str, default="", help="Optional valid-mask directory.")

    # Matching / naming
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="stem",
        choices=["stem", "name"],
        help="How to match directory files: by stem or exact filename.",
    )
    parser.add_argument(
        "--pred_suffix_to_strip",
        type=str,
        default="",
        help="Optional suffix stripped from prediction stem before matching, e.g. _albedo.",
    )
    parser.add_argument(
        "--gt_suffix_to_strip",
        type=str,
        default="",
        help="Optional suffix stripped from GT stem before matching.",
    )
    parser.add_argument(
        "--mask_suffix_to_strip",
        type=str,
        default="",
        help="Optional suffix stripped from mask stem before matching.",
    )

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV/JSON summaries.")
    parser.add_argument("--save_resized_preds", action="store_true", help="Save resized predictions for inspection.")

    # Evaluation options
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Recursively scan subdirectories for directory mode.",
    )
    parser.add_argument(
        "--no_recursive",
        dest="recursive",
        action="store_false",
        help="Do not scan subdirectories for directory mode.",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--resize_interp",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Interpolation used when resizing predictions to GT resolution.",
    )
    parser.add_argument(
        "--linearize_srgb",
        action="store_true",
        help="Convert both GT and prediction from sRGB to linear RGB before evaluation.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Threshold in [0,1] for binarizing mask images.",
    )
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        help="Compute LPIPS if torch and lpips are available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Device for LPIPS.",
    )

    return parser.parse_args()


def srgb_to_linear_np(img: np.ndarray) -> np.ndarray:
    """img in [0,1], shape HWC."""
    out = np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    )
    return out.astype(np.float32)


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


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_mask(path: Path, size: Tuple[int, int], threshold: float) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != size:
            img = img.resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return (arr >= threshold).astype(np.uint8)


def image_to_float_np(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def masked_mae(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(gt - pred).mean(axis=2)
    valid = mask > 0
    if valid.sum() == 0:
        return float("nan")
    return float(diff[valid].mean())


def masked_mse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    diff = ((gt - pred) ** 2).mean(axis=2)
    valid = mask > 0
    if valid.sum() == 0:
        return float("nan")
    return float(diff[valid].mean())


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / mse))


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    mse = masked_mse(gt, pred, mask)
    return psnr_from_mse(mse, data_range=data_range)


def compute_masked_errors(
    gt: np.ndarray, pred: np.ndarray, mask: np.ndarray
) -> Tuple[float, float, float, float, int]:
    diff = np.abs(gt - pred).mean(axis=2)
    diff_sq = ((gt - pred) ** 2).mean(axis=2)
    valid = mask > 0
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        return float("nan"), float("nan"), 0.0, 0.0, 0
    sum_abs = float(diff[valid].sum())
    sum_sq = float(diff_sq[valid].sum())
    mae = sum_abs / valid_pixels
    mse = sum_sq / valid_pixels
    return float(mae), float(mse), sum_abs, sum_sq, valid_pixels

def masked_ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Approximate masked SSIM by cropping to the valid bounding box, then filling invalid
    pixels with GT values so that outside-mask content does not affect the score.
    """
    if skimage_ssim is None:
        return float("nan")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float("nan")

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    gt_crop = gt[y0:y1, x0:x1].copy()
    pred_crop = pred[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]

    invalid = mask_crop == 0
    pred_crop[invalid] = gt_crop[invalid]

    if gt_crop.shape[0] < 7 or gt_crop.shape[1] < 7:
        return float("nan")

    return float(
        skimage_ssim(
            gt_crop,
            pred_crop,
            channel_axis=2,
            data_range=1.0,
        )
    )


class LPIPSEvaluator:
    def __init__(self, device: str = "cpu") -> None:
        self.available = torch is not None and lpips is not None
        self.device = device
        self.model = None
        if self.available:
            self.model = lpips.LPIPS(net="alex").to(device)
            self.model.eval()

    def __call__(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
        if not self.available:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        gt_crop = gt[y0:y1, x0:x1].copy()
        pred_crop = pred[y0:y1, x0:x1].copy()
        mask_crop = mask[y0:y1, x0:x1]

        invalid = mask_crop == 0
        pred_crop[invalid] = gt_crop[invalid]

        gt_t = torch.from_numpy(gt_crop).permute(2, 0, 1).unsqueeze(0).float()
        pred_t = torch.from_numpy(pred_crop).permute(2, 0, 1).unsqueeze(0).float()

        gt_t = gt_t * 2.0 - 1.0
        pred_t = pred_t * 2.0 - 1.0

        gt_t = gt_t.to(self.device)
        pred_t = pred_t.to(self.device)

        with torch.no_grad():
            val = self.model(gt_t, pred_t)
        return float(val.item())


def normalize_match_key(name: str, suffix_to_strip: str) -> str:
    stem = Path(name).stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def list_images(directory: Path, recursive: bool) -> List[Path]:
    if recursive:
        iterator = directory.rglob("*")
    else:
        iterator = directory.iterdir()
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS])


def build_pairs_from_dirs(
    gt_dir: Path,
    pred_dir: Path,
    mask_dir: Optional[Path],
    pair_mode: str,
    pred_suffix_to_strip: str,
    gt_suffix_to_strip: str,
    mask_suffix_to_strip: str,
    recursive: bool,
) -> List[Tuple[Path, Path, Optional[Path]]]:
    gt_files = list_images(gt_dir, recursive=recursive)
    pred_files = list_images(pred_dir, recursive=recursive)
    mask_files: List[Path] = []
    if mask_dir is not None:
        mask_files = list_images(mask_dir, recursive=recursive)

    def make_key(path: Path, root: Path, suffix_to_strip: str) -> str:
        rel = path.relative_to(root)
        if pair_mode == "name":
            return rel.as_posix()
        stem = normalize_match_key(path.name, suffix_to_strip)
        if rel.parent == Path("."):
            return stem
        return (rel.parent / stem).as_posix()

    if pair_mode == "name":
        gt_map = {make_key(p, gt_dir, gt_suffix_to_strip): p for p in gt_files}
        pred_map = {make_key(p, pred_dir, pred_suffix_to_strip): p for p in pred_files}
    else:
        gt_map = {make_key(p, gt_dir, gt_suffix_to_strip): p for p in gt_files}
        pred_map = {make_key(p, pred_dir, pred_suffix_to_strip): p for p in pred_files}
    mask_map: Dict[str, Path] = {}
    if mask_files:
        mask_map = {make_key(p, mask_dir, mask_suffix_to_strip): p for p in mask_files}

    common_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    pairs: List[Tuple[Path, Path, Optional[Path]]] = []

    for key in common_keys:
        gt_path = gt_map[key]
        pred_path = pred_map[key]
        mask_path = mask_map.get(key) if mask_map else None
        pairs.append((gt_path, pred_path, mask_path))

    return pairs


def evaluate_pair(
    gt_path: Path,
    pred_path: Path,
    mask_path: Optional[Path],
    args: argparse.Namespace,
    lpips_eval: Optional[LPIPSEvaluator],
    resized_pred_dir: Optional[Path],
    name: str,
) -> Tuple[ImageMetrics, float, float]:
    gt_img = load_rgb(gt_path)
    pred_img = load_rgb(pred_path)

    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, pil_interp_mode(args.resize_interp))

    if resized_pred_dir is not None:
        resized_pred_dir.mkdir(parents=True, exist_ok=True)
        pred_img.save(resized_pred_dir / pred_path.name)

    gt_np = image_to_float_np(gt_img)
    pred_np = image_to_float_np(pred_img)

    if args.linearize_srgb:
        gt_np = srgb_to_linear_np(gt_np)
        pred_np = srgb_to_linear_np(pred_np)

    if mask_path is not None and mask_path.exists():
        mask = load_mask(mask_path, gt_img.size, args.mask_threshold)
    else:
        mask = np.ones((gt_np.shape[0], gt_np.shape[1]), dtype=np.uint8)

    mae, mse, sum_abs, sum_sq, valid_pixels = compute_masked_errors(gt_np, pred_np, mask)
    psnr = psnr_from_mse(mse, data_range=1.0)
    ssim = masked_ssim(gt_np, pred_np, mask)
    lp = None
    if lpips_eval is not None:
        lp = lpips_eval(gt_np, pred_np, mask)

    return ImageMetrics(
        name=name,
        width=gt_img.size[0],
        height=gt_img.size[1],
        mae=mae,
        psnr=psnr,
        ssim=ssim,
        lpips=lp,
        valid_pixels=valid_pixels,
    ), sum_abs, sum_sq


def summarize(metrics: Sequence[ImageMetrics], total_abs: float, total_sq: float, total_valid: int) -> Dict[str, float]:
    def mean_valid(values: List[Optional[float]]) -> float:
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        if len(vals) == 0:
            return float("nan")
        return float(np.mean(vals))

    def median_valid(values: List[Optional[float]]) -> float:
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        if len(vals) == 0:
            return float("nan")
        return float(np.median(vals))

    def std_valid(values: List[Optional[float]]) -> float:
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        if len(vals) == 0:
            return float("nan")
        return float(np.std(vals))

    if total_valid > 0:
        mae_micro = total_abs / total_valid
        mse_micro = total_sq / total_valid
        psnr_micro = psnr_from_mse(mse_micro, data_range=1.0)
    else:
        mae_micro = float("nan")
        mse_micro = float("nan")
        psnr_micro = float("nan")

    return {
        "count": len(metrics),
        "valid_pixels_total": int(total_valid),
        "mae_mean": mean_valid([m.mae for m in metrics]),
        "psnr_mean": mean_valid([m.psnr for m in metrics]),
        "ssim_mean": mean_valid([m.ssim for m in metrics]),
        "lpips_mean": mean_valid([m.lpips for m in metrics]),
        "mae_micro": float(mae_micro),
        "mse_micro": float(mse_micro),
        "psnr_micro": float(psnr_micro),
        "ssim_median": median_valid([m.ssim for m in metrics]),
        "ssim_std": std_valid([m.ssim for m in metrics]),
        "lpips_median": median_valid([m.lpips for m in metrics]),
        "lpips_std": std_valid([m.lpips for m in metrics]),
    }


def save_results(output_dir: Path, metrics: Sequence[ImageMetrics], summary: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "albedo_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "width", "height", "mae", "psnr", "ssim", "lpips", "valid_pixels"],
        )
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))

    json_path = output_dir / "albedo_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    single_mode = bool(args.gt_path and args.pred_path)
    dir_mode = bool(args.gt_dir and args.pred_dir)

    if single_mode == dir_mode:
        raise ValueError("Use either single-pair mode (--gt_path/--pred_path) or directory mode (--gt_dir/--pred_dir).")

    output_dir = Path(args.output_dir)
    resized_pred_dir = output_dir / "resized_predictions" if args.save_resized_preds else None

    lpips_eval = None
    if args.compute_lpips:
        lpips_eval = LPIPSEvaluator(device=args.device)

    pairs: List[Tuple[Path, Path, Optional[Path]]] = []
    if single_mode:
        gt_path = Path(args.gt_path)
        pred_path = Path(args.pred_path)
        mask_path = Path(args.mask_path) if args.mask_path else None
        pairs = [(gt_path, pred_path, mask_path)]
    else:
        gt_dir = Path(args.gt_dir)
        pred_dir = Path(args.pred_dir)
        mask_dir = Path(args.mask_dir) if args.mask_dir else None
        pairs = build_pairs_from_dirs(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            mask_dir=mask_dir,
            pair_mode=args.pair_mode,
            pred_suffix_to_strip=args.pred_suffix_to_strip,
            gt_suffix_to_strip=args.gt_suffix_to_strip,
            mask_suffix_to_strip=args.mask_suffix_to_strip,
            recursive=args.recursive,
        )
        if len(pairs) == 0:
            raise RuntimeError("No matched GT/prediction pairs found.")

    metrics: List[ImageMetrics] = []
    total_abs = 0.0
    total_sq = 0.0
    total_valid = 0
    for gt_path, pred_path, mask_path in pairs:
        if single_mode:
            name = gt_path.stem
        else:
            name = gt_path.relative_to(gt_dir).as_posix()
        m, sum_abs, sum_sq = evaluate_pair(
            gt_path=gt_path,
            pred_path=pred_path,
            mask_path=mask_path,
            args=args,
            lpips_eval=lpips_eval,
            resized_pred_dir=resized_pred_dir,
            name=name,
        )
        metrics.append(m)
        total_abs += sum_abs
        total_sq += sum_sq
        total_valid += m.valid_pixels
        print(
            f"[{m.name}] "
            f"MAE={m.mae:.6f} | PSNR={m.psnr:.4f} | SSIM={m.ssim:.6f} | "
            f"LPIPS={m.lpips if m.lpips is not None else 'N/A'}"
        )

    summary = summarize(metrics, total_abs=total_abs, total_sq=total_sq, total_valid=total_valid)
    save_results(output_dir, metrics, summary)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
