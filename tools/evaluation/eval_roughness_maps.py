#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

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
    split: str
    width: int
    height: int
    mae: float
    rmse: float
    psnr: float
    ssim: float
    lpips: Optional[float]
    valid_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GT-only evaluation for predicted roughness maps against aligned ground-truth roughness, with optional masks."
    )

    parser.add_argument("--gt_path", type=str, default="", help="Single GT roughness path.")
    parser.add_argument("--pred_path", type=str, default="", help="Single predicted roughness path.")
    parser.add_argument("--mask_path", type=str, default="", help="Optional single valid-mask path.")

    parser.add_argument("--gt_dir", type=str, default="", help="GT roughness directory.")
    parser.add_argument("--pred_dir", type=str, default="", help="Prediction directory.")
    parser.add_argument("--mask_dir", type=str, default="", help="Optional valid-mask directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV/JSON summaries.")
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="stem",
        choices=["stem", "name"],
        help="How to match directory files: by stem or exact relative filename.",
    )
    parser.add_argument(
        "--pred_suffix_to_strip",
        type=str,
        default="_roughness",
        help="Suffix stripped from prediction stem before matching.",
    )
    parser.add_argument(
        "--gt_suffix_to_strip",
        type=str,
        default="_roughness",
        help="Suffix stripped from GT stem before matching.",
    )
    parser.add_argument(
        "--mask_suffix",
        type=str,
        default="_mask",
        help="Suffix stripped from mask stem before matching.",
    )
    parser.add_argument(
        "--pred_path_component_to_strip",
        type=str,
        default="",
        help="Optional relative path component removed from prediction keys before matching, e.g. 'roughness'.",
    )
    parser.add_argument(
        "--gt_path_component_to_strip",
        type=str,
        default="",
        help="Optional relative path component removed from GT keys before matching.",
    )
    parser.add_argument(
        "--mask_path_component_to_strip",
        type=str,
        default="",
        help="Optional relative path component removed from mask keys before matching.",
    )
    parser.add_argument("--save_resized_preds", action="store_true", help="Save resized predictions for inspection.")
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="Recursively search directories.")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false", help="Disable recursive search.")
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--resize_interp",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Interpolation used when resizing predictions to GT resolution.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Threshold in [0,1] for binarizing masks.",
    )
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        help="Compute LPIPS against aligned GT if torch and lpips are available. Gray roughness maps are repeated to 3 channels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Device used for LPIPS.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone used by LPIPS.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    single_mode = bool(args.gt_path and args.pred_path)
    dir_mode = bool(args.gt_dir and args.pred_dir)
    if single_mode == dir_mode:
        raise ValueError(
            "This script is GT-only. Use either single-pair mode (--gt_path/--pred_path) "
            "or directory mode (--gt_dir/--pred_dir)."
        )

    if single_mode:
        gt_path = Path(args.gt_path)
        pred_path = Path(args.pred_path)
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        if args.mask_path:
            mask_path = Path(args.mask_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
        return

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")


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


def list_images(directory: Path, recursive: bool) -> List[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS])


def normalize_match_key(name: str, suffix_to_strip: str) -> str:
    stem = Path(name).stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def load_gray(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L")


def image_to_float_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def split_from_name(name: str) -> str:
    for part in Path(name).parts:
        lower = part.lower()
        if "mainaxis" in lower:
            return "mainaxis"
        if "stresstest" in lower:
            return "stresstest"
    return "unknown"


def load_mask(path: Path, size: Tuple[int, int], threshold: float) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != size:
            img = img.resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr >= threshold).astype(np.uint8)


def is_same_existing_path(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.resolve() == path_b.resolve()
    except Exception:
        return path_a == path_b


def resolve_mask_path(
    gt_path: Path,
    mask_dir: Optional[Path],
    mask_map: Optional[Dict[str, Path]],
    pair_mode: str,
    gt_dir: Path,
    pair_key: str,
) -> Optional[Path]:
    if mask_dir is None:
        return None

    # First mirror eval_albedo_maps.py behavior: prefer exact GT filename,
    # then same stem with any supported image extension in the mask directory.
    rel = gt_path.relative_to(gt_dir)
    direct_candidate = mask_dir / rel
    if direct_candidate.exists() and not is_same_existing_path(direct_candidate, gt_path):
        return direct_candidate

    for ext in IMG_EXTS:
        probe = direct_candidate.with_suffix(ext)
        if probe.exists() and not is_same_existing_path(probe, gt_path):
            return probe

    # Fallback to roughness-specific suffix-aware matching.
    if mask_map:
        if pair_mode == "name":
            return mask_map.get(rel.as_posix()) or mask_map.get(pair_key)
        return mask_map.get(pair_key)

    return None


def build_pairs_from_dirs(
    gt_dir: Path,
    pred_dir: Path,
    mask_dir: Optional[Path],
    pair_mode: str,
    pred_suffix_to_strip: str,
    gt_suffix_to_strip: str,
    mask_suffix: str,
    pred_path_component_to_strip: str,
    gt_path_component_to_strip: str,
    mask_path_component_to_strip: str,
    recursive: bool,
) -> List[Tuple[Path, Path, Optional[Path]]]:
    gt_files = list_images(gt_dir, recursive)
    pred_files = list_images(pred_dir, recursive)
    mask_files = list_images(mask_dir, recursive) if mask_dir is not None else []

    def strip_path_component(rel: Path, component_to_strip: str) -> Path:
        if not component_to_strip:
            return rel
        return Path(*[part for part in rel.parts if part != component_to_strip])

    def make_key(path: Path, root: Path, suffix_to_strip: str, path_component_to_strip: str) -> str:
        rel = path.relative_to(root)
        rel = strip_path_component(rel, path_component_to_strip)
        if pair_mode == "name":
            return rel.as_posix()
        stem = normalize_match_key(path.name, suffix_to_strip)
        if rel.parent == Path("."):
            return stem
        return (rel.parent / stem).as_posix()

    gt_map = {make_key(p, gt_dir, gt_suffix_to_strip, gt_path_component_to_strip): p for p in gt_files}
    pred_map = {make_key(p, pred_dir, pred_suffix_to_strip, pred_path_component_to_strip): p for p in pred_files}
    mask_map = (
        {make_key(p, mask_dir, mask_suffix, mask_path_component_to_strip): p for p in mask_files}
        if mask_files
        else {}
    )
    common_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))

    pairs: List[Tuple[Path, Path, Optional[Path]]] = []
    for key in common_keys:
        gt_path = gt_map[key]
        pred_path = pred_map[key]
        mask_path = resolve_mask_path(
            gt_path=gt_path,
            mask_dir=mask_dir,
            mask_map=mask_map if mask_map else None,
            pair_mode=pair_mode,
            gt_dir=gt_dir,
            pair_key=key,
        )
        pairs.append((gt_path, pred_path, mask_path))
    return pairs


def masked_values(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = mask > 0
    return gt[valid], pred[valid]


def masked_mae(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    gt_v, pred_v = masked_values(gt, pred, mask)
    if gt_v.size == 0:
        return float("nan")
    return float(np.mean(np.abs(gt_v - pred_v)))


def masked_mse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    gt_v, pred_v = masked_values(gt, pred, mask)
    if gt_v.size == 0:
        return float("nan")
    return float(np.mean((gt_v - pred_v) ** 2))


def masked_rmse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    mse = masked_mse(gt, pred, mask)
    if not np.isfinite(mse):
        return float("nan")
    return float(math.sqrt(mse))


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    mse = masked_mse(gt, pred, mask)
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / mse))


def masked_ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
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

    return float(skimage_ssim(gt_crop, pred_crop, data_range=1.0))

class LPIPSEvaluator:
    def __init__(self, device: str = "cpu", net: str = "alex") -> None:
        self.available = torch is not None and lpips is not None
        self.device = device
        self.net = net
        self.model = None
        self.unavailable_reason: Optional[str] = None
        if not self.available:
            missing = []
            if torch is None:
                missing.append("torch")
            if lpips is None:
                missing.append("lpips")
            self.unavailable_reason = f"missing optional dependency: {', '.join(missing)}"
            return
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()

    def __call__(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
        if not self.available or self.model is None:
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

        gt_rgb = np.repeat(gt_crop[..., None], 3, axis=2)
        pred_rgb = np.repeat(pred_crop[..., None], 3, axis=2)

        gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).float()
        pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float()
        gt_t = gt_t * 2.0 - 1.0
        pred_t = pred_t * 2.0 - 1.0

        with torch.no_grad():
            value = self.model(gt_t.to(self.device), pred_t.to(self.device))
        return float(value.item())


def evaluate_pair(
    gt_path: Path,
    pred_path: Path,
    mask_path: Optional[Path],
    args: argparse.Namespace,
    lpips_eval: Optional[LPIPSEvaluator],
    resized_pred_dir: Optional[Path],
    name: str,
) -> ImageMetrics:
    gt_img = load_gray(gt_path)
    pred_img = load_gray(pred_path)
    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, pil_interp_mode(args.resize_interp))

    if resized_pred_dir is not None:
        resized_pred_dir.mkdir(parents=True, exist_ok=True)
        pred_img.save(resized_pred_dir / pred_path.name)

    gt = image_to_float_np(gt_img)
    pred = image_to_float_np(pred_img)
    if mask_path is not None and mask_path.exists():
        mask = load_mask(mask_path, gt_img.size, args.mask_threshold)
    else:
        mask = np.ones_like(gt, dtype=np.uint8)

    return ImageMetrics(
        name=name,
        split=split_from_name(name),
        width=gt_img.size[0],
        height=gt_img.size[1],
        mae=masked_mae(gt, pred, mask),
        rmse=masked_rmse(gt, pred, mask),
        psnr=masked_psnr(gt, pred, mask),
        ssim=masked_ssim(gt, pred, mask),
        lpips=lpips_eval(gt, pred, mask) if lpips_eval is not None else None,
        valid_pixels=int(mask.sum()),
    )


def summarize(metrics: Sequence[ImageMetrics]) -> Dict[str, float]:
    def mean_valid(values: Sequence[Optional[float]]) -> float:
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    def median_valid(values: Sequence[Optional[float]]) -> float:
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not vals:
            return float("nan")
        return float(np.median(vals))

    def std_valid(values: Sequence[Optional[float]]) -> float:
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not vals:
            return float("nan")
        return float(np.std(vals))

    return {
        "count": len(metrics),
        "mae_mean": mean_valid([m.mae for m in metrics]),
        "rmse_mean": mean_valid([m.rmse for m in metrics]),
        "psnr_mean": mean_valid([m.psnr for m in metrics]),
        "ssim_mean": mean_valid([m.ssim for m in metrics]),
        "lpips_mean": mean_valid([m.lpips for m in metrics]),
        "mae_median": median_valid([m.mae for m in metrics]),
        "rmse_median": median_valid([m.rmse for m in metrics]),
        "psnr_median": median_valid([m.psnr for m in metrics]),
        "ssim_median": median_valid([m.ssim for m in metrics]),
        "lpips_median": median_valid([m.lpips for m in metrics]),
        "mae_std": std_valid([m.mae for m in metrics]),
        "rmse_std": std_valid([m.rmse for m in metrics]),
        "psnr_std": std_valid([m.psnr for m in metrics]),
        "ssim_std": std_valid([m.ssim for m in metrics]),
        "lpips_std": std_valid([m.lpips for m in metrics]),
    }


def summarize_by_split(metrics: Sequence[ImageMetrics]) -> Dict[str, Dict[str, float]]:
    return {
        split_name: summarize([m for m in metrics if m.split == split_name])
        for split_name in ("mainaxis", "stresstest")
    }


def save_results(output_dir: Path, metrics: Sequence[ImageMetrics], summary: Dict[str, object]) -> None:
    def sanitize_json_value(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: sanitize_json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize_json_value(item) for item in value]
        return value

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "roughness_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "split",
                "width",
                "height",
                "mae",
                "rmse",
                "psnr",
                "ssim",
                "lpips",
                "valid_pixels",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(asdict(metric))

    json_path = output_dir / "roughness_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {key: sanitize_json_value(value) for key, value in summary.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_results_by_split(output_dir: Path, metrics: Sequence[ImageMetrics]) -> None:
    for split_name in ("mainaxis", "stresstest"):
        split_metrics = [m for m in metrics if m.split == split_name]
        if not split_metrics:
            continue
        save_results(output_dir / split_name, split_metrics, summarize(split_metrics))


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    single_mode = bool(args.gt_path and args.pred_path)

    output_dir = Path(args.output_dir)
    resized_pred_dir = output_dir / "resized_predictions" if args.save_resized_preds else None
    if skimage_ssim is None:
        print("[warn] SSIM disabled: missing optional dependency skimage.", file=sys.stderr)
    lpips_eval = None
    if args.compute_lpips:
        lpips_eval = LPIPSEvaluator(device=args.device, net=args.lpips_net)
        if not lpips_eval.available:
            print(f"[warn] LPIPS disabled: {lpips_eval.unavailable_reason}", file=sys.stderr)

    if single_mode:
        gt_dir = None
        pairs = [(Path(args.gt_path), Path(args.pred_path), Path(args.mask_path) if args.mask_path else None)]
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
            mask_suffix=args.mask_suffix,
            pred_path_component_to_strip=args.pred_path_component_to_strip,
            gt_path_component_to_strip=args.gt_path_component_to_strip,
            mask_path_component_to_strip=args.mask_path_component_to_strip,
            recursive=args.recursive,
        )
        if not pairs:
            raise RuntimeError("No matched GT/prediction pairs found.")

    metrics: List[ImageMetrics] = []
    for gt_path, pred_path, mask_path in pairs:
        name = gt_path.stem if single_mode else gt_path.relative_to(gt_dir).as_posix()
        metric = evaluate_pair(
            gt_path=gt_path,
            pred_path=pred_path,
            mask_path=mask_path,
            args=args,
            lpips_eval=lpips_eval,
            resized_pred_dir=resized_pred_dir,
            name=name,
        )
        metrics.append(metric)
        print(
            f"[{metric.name}] "
            f"MAE={metric.mae:.6f} | RMSE={metric.rmse:.6f} | PSNR={metric.psnr:.6f} | "
            f"SSIM={metric.ssim:.6f} | LPIPS={metric.lpips if metric.lpips is not None else 'N/A'}"
        )

    summary: Dict[str, object] = summarize(metrics)
    summary["by_split"] = summarize_by_split(metrics)
    save_results(output_dir, metrics, summary)
    save_results_by_split(output_dir, metrics)

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
