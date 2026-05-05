#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim
import torch
import lpips


DEFAULT_GT_ROOT = Path("data/metallic/gt")
DEFAULT_PRED_ROOT = Path("outputs/evaluation/metallic/predictions")
STRESSTEST_LIGHTING_GROUPS = [
    ("L0", "normal_light"),
    ("L1", "Low-light"),
    ("L2", "HDR"),
    ("L3", "dark heavy"),
    ("L4", "Highlight-heavy"),
    ("L5", "Mix-temperature lighting"),
]


@dataclass
class ImageMetrics:
    name: str
    split: str
    lighting: str
    gt_path: str
    pred_path: str
    mask_path: str
    width: int
    height: int
    total_pixels: int
    valid_pixels: int
    mask_coverage: float

    # Main full-map continuous regression metrics.
    mae: float
    rmse: float
    psnr: float

    # Auxiliary visual/structural metrics for metallic maps.
    # These are useful diagnostics, but should not be interpreted as the
    # primary physical correctness metrics for metalness.
    ssim: float
    lpips: Optional[float]

    # Continuous-metallicity weighted metrics.
    # metal weight = gt, nonmetal weight = 1 - gt.
    gt_metallic_sum: float
    gt_nonmetallic_sum: float
    metal_weighted_abs_sum: float
    nonmetal_weighted_abs_sum: float
    gt_metallic_mean: float
    pred_metallic_mean: float
    metal_weighted_mae: float
    nonmetal_weighted_mae: float

    # Bias diagnostics: tells whether the model tends to over-predict or
    # under-predict metallic values.
    signed_error_sum: float
    over_metal_error_sum: float
    under_metal_error_sum: float
    signed_error_mean: float
    over_metal_error_mean: float
    under_metal_error_mean: float


class LPIPSEvaluator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()

    def __call__(self, gt_gray: np.ndarray, pred_gray: np.ndarray) -> Optional[float]:
        gt_rgb = np.repeat(gt_gray[..., None], 3, axis=2)
        pred_rgb = np.repeat(pred_gray[..., None], 3, axis=2)

        gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float()
        pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float()
        gt_t = gt_t * 2.0 - 1.0
        pred_t = pred_t * 2.0 - 1.0
        gt_t = gt_t.to(self.device)
        pred_t = pred_t.to(self.device)

        with torch.no_grad():
            value = self.model(gt_t, pred_t)
        return float(value.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted metallic maps against continuous GT metallic maps."
    )
    parser.add_argument("--gt_root", type=Path, default=DEFAULT_GT_ROOT, help="GT root directory.")
    parser.add_argument("--pred_root", type=Path, default=DEFAULT_PRED_ROOT, help="Prediction root directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save CSV/JSON results.")
    parser.add_argument(
        "--resize_interp",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help=(
            "Interpolation used when resizing predictions to GT resolution. "
            "Default is bilinear because metallic is evaluated as a continuous map, "
            "not as a hard segmentation mask."
        ),
    )
    parser.add_argument(
        "--min_effective_metal_weight",
        type=float,
        default=10.0,
        help=(
            "Minimum sum(gt) required for image-level metal_weighted_mae. "
            "Images below this value are treated as NaN for image-level statistics, "
            "but still contribute to the micro weighted metric."
        ),
    )
    parser.add_argument(
        "--min_effective_nonmetal_weight",
        type=float,
        default=10.0,
        help=(
            "Minimum sum(1 - gt) required for image-level nonmetal_weighted_mae. "
            "Images below this value are treated as NaN for image-level statistics, "
            "but still contribute to the micro weighted metric."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for LPIPS.",
    )
    parser.add_argument(
        "--skip_lpips",
        action="store_true",
        help=(
            "Skip LPIPS computation. LPIPS is kept as an auxiliary visual metric "
            "because metallic maps are physical parameter maps rather than natural images."
        ),
    )
    parser.add_argument(
        "--save_resized_preds",
        action="store_true",
        help="Save resized predictions next to the outputs for inspection.",
    )
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


def load_gray(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L")


def image_to_float_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / mse))


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def metric_values(values: Sequence[Optional[float]]) -> List[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def metric_mean(values: Sequence[Optional[float]]) -> float:
    vals = metric_values(values)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def metric_median(values: Sequence[Optional[float]]) -> float:
    vals = metric_values(values)
    if not vals:
        return float("nan")
    return float(np.median(vals))


def metric_std(values: Sequence[Optional[float]]) -> float:
    vals = metric_values(values)
    if not vals:
        return float("nan")
    return float(np.std(vals))


def metric_count(values: Sequence[Optional[float]]) -> int:
    return len(metric_values(values))


def build_pred_to_gt_path(pred_path: Path, pred_root: Path, gt_root: Path) -> Path:
    rel = pred_path.relative_to(pred_root)
    parts = rel.parts

    if len(parts) >= 8 and parts[-2] == "metallic":
        camera_dir = parts[-3]
        scene_prefix = parts[:-4]
    elif len(parts) >= 7 and parts[-3] == "Image":
        camera_dir = parts[-2]
        scene_prefix = parts[:-3]
    else:
        raise ValueError(f"Unexpected prediction path layout: {pred_path}")

    gt_path = gt_root.joinpath(*scene_prefix, "Metallic", camera_dir, "Metallic_0001.png")
    return gt_path


def build_gt_to_mask_path(gt_path: Path) -> Path:
    if gt_path.parent.parent.name != "Metallic":
        raise ValueError(f"Unexpected GT path layout: {gt_path}")

    view_dir = gt_path.parent.parent.parent
    camera_dir = gt_path.parent.name
    return view_dir / "MetallicEvalMask" / camera_dir / "MetallicEvalMask_0001.png"


def lighting_from_name(name: str, split: str) -> str:
    if split != "stresstest":
        return ""

    for part in Path(name).parts:
        match = re.search(r"(?:^|_)L([0-5])(?:_|$)", part)
        if match:
            return f"L{match.group(1)}"
    return "unknown"


def is_prediction_metallic_path(path: Path) -> bool:
    if path.suffix.lower() != ".png":
        return False

    stem = path.stem
    if path.parent.name == "metallic":
        return stem.startswith("Image_") and stem.endswith("_metallic")

    if path.parent.parent.name != "Image":
        return False

    if stem.endswith("_metallic"):
        return True

    non_metallic_suffixes = ("_albedo", "_depth", "_normal", "_roughness")
    return stem.startswith("Image_") and not stem.endswith(non_metallic_suffixes)


def collect_pairs(pred_root: Path, gt_root: Path) -> List[Tuple[Path, Path, Path, str, str]]:
    pred_files = sorted(path for path in pred_root.rglob("*.png") if is_prediction_metallic_path(path))
    pairs: List[Tuple[Path, Path, Path, str, str]] = []
    missing_gt: List[Path] = []
    missing_mask: List[Path] = []

    for pred_path in pred_files:
        gt_path = build_pred_to_gt_path(pred_path, pred_root, gt_root)
        if not gt_path.exists():
            missing_gt.append(gt_path)
            continue
        mask_path = build_gt_to_mask_path(gt_path)
        if not mask_path.exists():
            missing_mask.append(mask_path)
            continue

        rel = gt_path.relative_to(gt_root)
        name = rel.as_posix()
        split = rel.parts[0]
        pairs.append((gt_path, pred_path, mask_path, name, split))

    if missing_gt:
        print(f"Warning: {len(missing_gt)} prediction files did not find GT. First missing GT: {missing_gt[0]}")
    if missing_mask:
        print(f"Warning: {len(missing_mask)} prediction files did not find mask. First missing mask: {missing_mask[0]}")

    return pairs


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    h, w = gt.shape
    if min(h, w) < 7:
        return float("nan")
    return float(skimage_ssim(gt, pred, data_range=1.0))


def weighted_mae_or_nan(weighted_abs_sum: float, weight_sum: float, min_effective_weight: float) -> float:
    if weight_sum < min_effective_weight:
        return float("nan")
    return safe_div(weighted_abs_sum, weight_sum)


def load_eval_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    mask_img = load_gray(mask_path)
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
    return image_to_float_np(mask_img) > 0.5


def evaluate_pair(
    gt_path: Path,
    pred_path: Path,
    mask_path: Path,
    name: str,
    split: str,
    pred_root: Path,
    resize_interp: str,
    min_effective_metal_weight: float,
    min_effective_nonmetal_weight: float,
    lpips_eval: Optional[LPIPSEvaluator],
    resized_pred_dir: Optional[Path],
) -> Tuple[ImageMetrics, float, float, int, float, float, float]:
    gt_img = load_gray(gt_path)
    pred_img = load_gray(pred_path)

    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, pil_interp_mode(resize_interp))

    if resized_pred_dir is not None:
        resized_target = resized_pred_dir / pred_path.relative_to(pred_root)
        resized_target.parent.mkdir(parents=True, exist_ok=True)
        pred_img.save(resized_target)

    gt = image_to_float_np(gt_img)
    pred = image_to_float_np(pred_img)
    mask = load_eval_mask(mask_path, gt_img.size)
    total_pixels = int(gt.size)
    valid_pixels = int(mask.sum())
    if valid_pixels <= 0:
        raise ValueError(f"Mask contains no valid pixels: {mask_path}")

    gt_valid = gt[mask]
    pred_valid = pred[mask]

    # Masked continuous regression over valid rendered/evaluable pixels.
    diff = np.abs(gt_valid - pred_valid)
    diff_sq = (gt_valid - pred_valid) ** 2

    sum_abs = float(diff.sum())
    sum_sq = float(diff_sq.sum())
    mse = safe_div(sum_sq, valid_pixels)
    mae = safe_div(sum_abs, valid_pixels)
    rmse = float(math.sqrt(mse)) if np.isfinite(mse) else float("nan")
    psnr = psnr_from_mse(mse)
    gt_masked = np.where(mask, gt, 0.0)
    pred_masked = np.where(mask, pred, 0.0)
    ssim = compute_ssim(gt_masked, pred_masked)
    lp = lpips_eval(gt_masked, pred_masked) if lpips_eval is not None else None

    # Scheme 3: continuous GT weighted metallic metrics.
    # Do not binarize gt. Pixels with higher gt metallic contribute more to
    # metal_weighted_mae; pixels with lower gt metallic contribute more to
    # nonmetal_weighted_mae.
    metal_weight = gt_valid
    nonmetal_weight = 1.0 - gt_valid

    gt_metallic_sum = float(metal_weight.sum())
    gt_nonmetallic_sum = float(nonmetal_weight.sum())
    metal_weighted_abs_sum = float((metal_weight * diff).sum())
    nonmetal_weighted_abs_sum = float((nonmetal_weight * diff).sum())

    metal_weighted_mae = weighted_mae_or_nan(
        metal_weighted_abs_sum,
        gt_metallic_sum,
        min_effective_weight=min_effective_metal_weight,
    )
    nonmetal_weighted_mae = weighted_mae_or_nan(
        nonmetal_weighted_abs_sum,
        gt_nonmetallic_sum,
        min_effective_weight=min_effective_nonmetal_weight,
    )

    # Image-level average GT metallic value, not a binary metal-pixel ratio.
    gt_metallic_mean = safe_div(gt_metallic_sum, valid_pixels)
    pred_metallic_mean = float(pred_valid.mean())

    signed_error = pred_valid - gt_valid
    over_metal_error = np.maximum(signed_error, 0.0)
    under_metal_error = np.maximum(-signed_error, 0.0)

    signed_error_sum = float(signed_error.sum())
    over_metal_error_sum = float(over_metal_error.sum())
    under_metal_error_sum = float(under_metal_error.sum())

    signed_error_mean = safe_div(signed_error_sum, valid_pixels)
    over_metal_error_mean = safe_div(over_metal_error_sum, valid_pixels)
    under_metal_error_mean = safe_div(under_metal_error_sum, valid_pixels)

    metrics = ImageMetrics(
        name=name,
        split=split,
        lighting=lighting_from_name(name, split),
        gt_path=str(gt_path),
        pred_path=str(pred_path),
        mask_path=str(mask_path),
        width=gt_img.size[0],
        height=gt_img.size[1],
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        mask_coverage=safe_div(valid_pixels, total_pixels),
        mae=mae,
        rmse=rmse,
        psnr=psnr,
        ssim=ssim,
        lpips=lp,
        gt_metallic_sum=gt_metallic_sum,
        gt_nonmetallic_sum=gt_nonmetallic_sum,
        metal_weighted_abs_sum=metal_weighted_abs_sum,
        nonmetal_weighted_abs_sum=nonmetal_weighted_abs_sum,
        gt_metallic_mean=gt_metallic_mean,
        pred_metallic_mean=pred_metallic_mean,
        metal_weighted_mae=metal_weighted_mae,
        nonmetal_weighted_mae=nonmetal_weighted_mae,
        signed_error_sum=signed_error_sum,
        over_metal_error_sum=over_metal_error_sum,
        under_metal_error_sum=under_metal_error_sum,
        signed_error_mean=signed_error_mean,
        over_metal_error_mean=over_metal_error_mean,
        under_metal_error_mean=under_metal_error_mean,
    )
    return (
        metrics,
        sum_abs,
        sum_sq,
        valid_pixels,
        signed_error_sum,
        over_metal_error_sum,
        under_metal_error_sum,
    )


def summarize_from_sums(
    metrics: Sequence[ImageMetrics],
    total_abs: float,
    total_sq: float,
    total_valid: int,
    total_metal_weighted_abs: float,
    total_gt_metallic_sum: float,
    total_nonmetal_weighted_abs: float,
    total_gt_nonmetallic_sum: float,
    total_signed_error: float,
    total_over_metal_error: float,
    total_under_metal_error: float,
) -> dict:
    return {
        "count": len(metrics),
        "valid_pixels_total": int(total_valid),

        "gt_metallic_mean": metric_mean([m.gt_metallic_mean for m in metrics]),
        "pred_metallic_mean": metric_mean([m.pred_metallic_mean for m in metrics]),

        "mae_mean": metric_mean([m.mae for m in metrics]),
        "mae_median": metric_median([m.mae for m in metrics]),
        "mae_std": metric_std([m.mae for m in metrics]),

        "rmse_mean": metric_mean([m.rmse for m in metrics]),

        "psnr_mean": metric_mean([m.psnr for m in metrics]),

        "ssim_mean": metric_mean([m.ssim for m in metrics]),

        "lpips_mean": metric_mean([m.lpips for m in metrics]),

        "metal_weighted_mae_mean": metric_mean([m.metal_weighted_mae for m in metrics]),
        "metal_weighted_mae_micro": safe_div(total_metal_weighted_abs, total_gt_metallic_sum),

        "nonmetal_weighted_mae_mean": metric_mean([m.nonmetal_weighted_mae for m in metrics]),
        "nonmetal_weighted_mae_micro": safe_div(total_nonmetal_weighted_abs, total_gt_nonmetallic_sum),

        "over_metal_error_micro": safe_div(total_over_metal_error, total_valid),
        "under_metal_error_micro": safe_div(total_under_metal_error, total_valid),

        "num_valid_metal_weighted_images": metric_count([m.metal_weighted_mae for m in metrics]),
        "num_valid_nonmetal_weighted_images": metric_count([m.nonmetal_weighted_mae for m in metrics]),
    }


def summarize(metrics: Sequence[ImageMetrics]) -> dict:
    total_abs = float(sum(m.mae * m.valid_pixels for m in metrics if np.isfinite(m.mae)))
    total_sq = float(sum((m.rmse ** 2) * m.valid_pixels for m in metrics if np.isfinite(m.rmse)))
    total_valid = int(sum(m.valid_pixels for m in metrics))
    total_metal_weighted_abs = float(sum(m.metal_weighted_abs_sum for m in metrics))
    total_gt_metallic_sum = float(sum(m.gt_metallic_sum for m in metrics))
    total_nonmetal_weighted_abs = float(sum(m.nonmetal_weighted_abs_sum for m in metrics))
    total_gt_nonmetallic_sum = float(sum(m.gt_nonmetallic_sum for m in metrics))
    total_signed_error = float(sum(m.signed_error_sum for m in metrics))
    total_over_metal_error = float(sum(m.over_metal_error_sum for m in metrics))
    total_under_metal_error = float(sum(m.under_metal_error_sum for m in metrics))

    return summarize_from_sums(
        metrics=metrics,
        total_abs=total_abs,
        total_sq=total_sq,
        total_valid=total_valid,
        total_metal_weighted_abs=total_metal_weighted_abs,
        total_gt_metallic_sum=total_gt_metallic_sum,
        total_nonmetal_weighted_abs=total_nonmetal_weighted_abs,
        total_gt_nonmetallic_sum=total_gt_nonmetallic_sum,
        total_signed_error=total_signed_error,
        total_over_metal_error=total_over_metal_error,
        total_under_metal_error=total_under_metal_error,
    )


def summarize_by_split(metrics: Sequence[ImageMetrics]) -> dict:
    grouped: dict = {}

    mainaxis_metrics = [m for m in metrics if m.split == "mainaxis"]
    grouped["mainaxis"] = summarize(mainaxis_metrics)

    stresstest_metrics = [m for m in metrics if m.split == "stresstest"]
    grouped["stresstest"] = {
        "overall": summarize(stresstest_metrics),
        "by_lighting": {
            f"{code}_{label}": summarize([m for m in stresstest_metrics if m.lighting == code])
            for code, label in STRESSTEST_LIGHTING_GROUPS
        },
    }
    unknown_lighting_metrics = [m for m in stresstest_metrics if m.lighting == "unknown"]
    if unknown_lighting_metrics:
        grouped["stresstest"]["by_lighting"]["unknown"] = summarize(unknown_lighting_metrics)

    return grouped



def metric_groups_metadata() -> dict:
    return {
        "main_continuous_regression": ["mae", "rmse", "psnr"],
        "weighted_metallicity_diagnostics": [
            "metal_weighted_mae",
            "nonmetal_weighted_mae",
            "gt_metallic_mean",
            "pred_metallic_mean",
        ],
        "bias_diagnostics": [
            "signed_error",
            "over_metal_error",
            "under_metal_error",
        ],
        "auxiliary_visual_metrics": ["ssim", "lpips"],
        "notes": {
            "ssim_lpips": (
                "SSIM and LPIPS are reported as auxiliary visual/structural diagnostics. "
                "They should not be treated as primary physical-correctness metrics for metallic maps."
            ),
            "gt_metallic_mean": (
                "Average continuous GT metallic value in [0,1]; this is not a binary metal-pixel ratio."
            ),
            "stresstest_lighting_groups": {
                f"{code}_{label}": label for code, label in STRESSTEST_LIGHTING_GROUPS
            },
            "resize_interp_default": (
                "Default resize interpolation is bilinear because metallic is evaluated as a continuous map."
            ),
            "mask": (
                "Metrics are computed over MetallicEvalMask valid pixels. "
                "SSIM/LPIPS are auxiliary diagnostics computed after zeroing pixels outside the mask."
            ),
        },
    }

def save_results(output_dir: Path, metrics: Sequence[ImageMetrics], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metallic_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))

    json_path = output_dir / "metallic_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_results_by_split(output_root: Path, pred_root: Path, metrics: Sequence[ImageMetrics]) -> None:
    prefix = pred_root.name
    metric_groups = metric_groups_metadata()

    mainaxis_metrics = [m for m in metrics if m.split == "mainaxis"]
    if mainaxis_metrics:
        mainaxis_summary = summarize(mainaxis_metrics)
        mainaxis_summary["metric_groups"] = metric_groups
        save_results(output_root / f"{prefix}_mainaxis", mainaxis_metrics, mainaxis_summary)

    stresstest_metrics = [m for m in metrics if m.split == "stresstest"]
    if not stresstest_metrics:
        return

    stresstest_summary = {
        "overall": summarize(stresstest_metrics),
        "by_lighting": {
            f"{code}_{label}": summarize([m for m in stresstest_metrics if m.lighting == code])
            for code, label in STRESSTEST_LIGHTING_GROUPS
        },
        "metric_groups": metric_groups,
    }
    unknown_lighting_metrics = [m for m in stresstest_metrics if m.lighting == "unknown"]
    if unknown_lighting_metrics:
        stresstest_summary["by_lighting"]["unknown"] = summarize(unknown_lighting_metrics)

    save_results(output_root / f"{prefix}_stresstest", stresstest_metrics, stresstest_summary)


def main() -> None:
    args = parse_args()

    pairs = collect_pairs(pred_root=args.pred_root, gt_root=args.gt_root)
    if not pairs:
        raise RuntimeError("No matched GT/prediction pairs found.")

    lpips_eval = None if args.skip_lpips else LPIPSEvaluator(device=args.device)
    resized_pred_dir = args.output_dir / "resized_predictions" if args.save_resized_preds else None

    metrics: List[ImageMetrics] = []
    total_abs = 0.0
    total_sq = 0.0
    total_valid = 0
    total_signed_error = 0.0
    total_over_metal_error = 0.0
    total_under_metal_error = 0.0

    for gt_path, pred_path, mask_path, name, split in pairs:
        (
            item,
            sum_abs,
            sum_sq,
            valid_pixels,
            signed_error_sum,
            over_metal_error_sum,
            under_metal_error_sum,
        ) = evaluate_pair(
            gt_path=gt_path,
            pred_path=pred_path,
            mask_path=mask_path,
            name=name,
            split=split,
            pred_root=args.pred_root,
            resize_interp=args.resize_interp,
            min_effective_metal_weight=args.min_effective_metal_weight,
            min_effective_nonmetal_weight=args.min_effective_nonmetal_weight,
            lpips_eval=lpips_eval,
            resized_pred_dir=resized_pred_dir,
        )
        metrics.append(item)
        total_abs += sum_abs
        total_sq += sum_sq
        total_valid += valid_pixels
        total_signed_error += signed_error_sum
        total_over_metal_error += over_metal_error_sum
        total_under_metal_error += under_metal_error_sum
        print(
            f"[{item.name}] "
            f"MAE={item.mae:.6f} | RMSE={item.rmse:.6f} | PSNR={item.psnr:.4f} | "
            f"SSIM={item.ssim:.6f} | "
            f"Metal-wMAE={item.metal_weighted_mae:.6f} | "
            f"Nonmetal-wMAE={item.nonmetal_weighted_mae:.6f} | "
            f"GTMetalMean={item.gt_metallic_mean:.6f} | "
            f"PredMetalMean={item.pred_metallic_mean:.6f} | "
            f"SignedErr={item.signed_error_mean:.6f} | "
            f"OverMetal={item.over_metal_error_mean:.6f} | "
            f"UnderMetal={item.under_metal_error_mean:.6f} | "
            f"LPIPS={item.lpips if item.lpips is not None else 'N/A'}"
        )

    overall_summary = summarize_from_sums(
        metrics=metrics,
        total_abs=total_abs,
        total_sq=total_sq,
        total_valid=total_valid,
        total_metal_weighted_abs=float(sum(m.metal_weighted_abs_sum for m in metrics)),
        total_gt_metallic_sum=float(sum(m.gt_metallic_sum for m in metrics)),
        total_nonmetal_weighted_abs=float(sum(m.nonmetal_weighted_abs_sum for m in metrics)),
        total_gt_nonmetallic_sum=float(sum(m.gt_nonmetallic_sum for m in metrics)),
        total_signed_error=total_signed_error,
        total_over_metal_error=total_over_metal_error,
        total_under_metal_error=total_under_metal_error,
    )
    by_split_summary = summarize_by_split(metrics)

    save_results_by_split(args.output_dir, args.pred_root, metrics)

    print("\n=== Overall Summary ===")
    print(json.dumps(overall_summary, indent=2, ensure_ascii=False))
    print("\n=== Split Summary ===")
    print(json.dumps(by_split_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
