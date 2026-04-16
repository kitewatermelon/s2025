"""
tools/hu_analysis.py — HU (Hounsfield Unit) 범위 분석 + 예측 vs GT 비교 (R2-9)

Sub-commands:
  data       원본 데이터셋 HU 분포 분석 → CSV
  compare    모델 예측 CT vs GT CT 비교 (tissue별 MAE, histogram)

Usage:
    # 데이터 분포 분석
    python tools/hu_analysis.py data --data_path synthRAD2025_Train_2.5D/Task2

    # 예측 vs GT 비교 (모델 추론 결과 .mha 폴더 기준)
    python tools/hu_analysis.py compare \
        --pred_dir results/infer_2d \
        --gt_dir   synthRAD2025_Train_2.5D/Task2 \
        --output   results/hu_compare.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ── HU 범위 정의 ──────────────────────────────────────────────────────────────

HU_MIN = -1024.0
HU_MAX = 3071.0

TISSUE_RANGES = {
    "air":        (-1024, -900),
    "lung":       (-900,  -500),
    "soft_tissue": (-100,   100),
    "bone":        (200,  3071),
}


def load_volume_hu(path: str) -> np.ndarray:
    """Load a .mha/.nii file and return the raw HU array (D, H, W)."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


# ── sub-command: data ─────────────────────────────────────────────────────────

def analyze_volume(path: str) -> dict:
    """Return HU statistics for a single volume."""
    arr = load_volume_hu(path)
    masked = arr[arr > HU_MIN]
    return {
        "path": str(path),
        "min_hu": float(arr.min()),
        "max_hu": float(arr.max()),
        "mean_hu": float(arr.mean()),
        "std_hu": float(arr.std()),
        "p5_hu": float(np.percentile(arr, 5)),
        "p95_hu": float(np.percentile(arr, 95)),
        "masked_mean_hu": float(masked.mean()) if masked.size > 0 else float("nan"),
        "masked_std_hu": float(masked.std()) if masked.size > 0 else float("nan"),
    }


def collect_volumes(data_path: str, anatomy: list, modality: list) -> list:
    root = Path(data_path)
    paths = []
    for anat in anatomy:
        anat_dir = root / anat.upper()
        if not anat_dir.exists():
            print(f"[WARN] {anat_dir} not found, skipping.")
            continue
        for case_dir in sorted(anat_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            for mod in modality:
                vol_path = case_dir / f"{mod}.mha"
                if vol_path.exists():
                    paths.append((anat.upper(), mod, str(vol_path)))
    return paths


def cmd_data(args):
    volume_list = collect_volumes(args.data_path, args.anatomy, args.modality)
    if not volume_list:
        print("No volumes found. Check --data_path and --anatomy.")
        return

    print(f"Found {len(volume_list)} volumes. Analyzing...")
    records = []
    for anat, mod, path in tqdm(volume_list, desc="HU analysis"):
        stats = analyze_volume(path)
        stats["anatomy"] = anat
        stats["modality"] = mod
        records.append(stats)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "anatomy", "modality", "path",
        "min_hu", "max_hu", "mean_hu", "std_hu",
        "p5_hu", "p95_hu", "masked_mean_hu", "masked_std_hu",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved: {out_path}")

    print("\n--- Summary (mean ± std HU) ---")
    groups: dict = defaultdict(list)
    for r in records:
        groups[(r["anatomy"], r["modality"])].append(r["mean_hu"])
    for (anat, mod), vals in sorted(groups.items()):
        arr = np.array(vals)
        print(f"  {anat}/{mod}: {arr.mean():.1f} ± {arr.std():.1f} HU  (n={len(vals)})")


# ── sub-command: compare ──────────────────────────────────────────────────────

def tissue_mae(pred: np.ndarray, gt: np.ndarray, hu_lo: float, hu_hi: float) -> float:
    """MAE within a tissue HU range (defined on GT mask)."""
    mask = (gt >= hu_lo) & (gt <= hu_hi)
    if mask.sum() == 0:
        return float("nan")
    return float(np.abs(pred[mask] - gt[mask]).mean())


def compare_pair(pred_path: str, gt_path: str) -> dict:
    """Compare one predicted volume against its GT. Returns per-tissue MAE + overall RMSE."""
    pred = load_volume_hu(pred_path)
    gt   = load_volume_hu(gt_path)

    # Clip to valid HU range before comparison
    pred = np.clip(pred, HU_MIN, HU_MAX)
    gt   = np.clip(gt,   HU_MIN, HU_MAX)

    result = {
        "pred_path": str(pred_path),
        "gt_path":   str(gt_path),
        "rmse":      float(np.sqrt(np.mean((pred - gt) ** 2))),
        "mae":       float(np.abs(pred - gt).mean()),
    }
    for tissue, (lo, hi) in TISSUE_RANGES.items():
        result[f"mae_{tissue}"] = tissue_mae(pred, gt, lo, hi)

    return result


def plot_hu_histogram(pred: np.ndarray, gt: np.ndarray, title: str, out_path: str) -> None:
    """3-panel: GT / Pred / Error histogram."""
    bins = np.linspace(HU_MIN, HU_MAX, 200)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(gt.ravel(), bins=bins, color="steelblue", alpha=0.8, density=True)
    axes[0].set_title("GT CT")
    axes[0].set_xlabel("HU")
    axes[0].set_ylabel("Density")

    axes[1].hist(pred.ravel(), bins=bins, color="tomato", alpha=0.8, density=True)
    axes[1].set_title("Predicted CT")
    axes[1].set_xlabel("HU")

    error = pred - gt
    err_bins = np.linspace(-500, 500, 100)
    axes[2].hist(error.ravel(), bins=err_bins, color="purple", alpha=0.8, density=True)
    axes[2].set_title("Error (Pred - GT)")
    axes[2].set_xlabel("ΔHU")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)

    fig.suptitle(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved: {out_path}")


def cmd_compare(args):
    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir)

    pred_files = sorted(pred_dir.glob("**/*.mha"))
    if not pred_files:
        print(f"No .mha files found in {pred_dir}")
        return

    records = []
    for pred_path in tqdm(pred_files, desc="Comparing HU"):
        # GT 파일: gt_dir/<anatomy>/<subj_id>/ct.mha 또는 same structure
        subj_id  = pred_path.stem          # SaveImage postfix = subj_id
        gt_candidates = list(gt_dir.glob(f"**/{subj_id}/ct.mha"))
        if not gt_candidates:
            print(f"[SKIP] GT not found for {subj_id}")
            continue
        gt_path = gt_candidates[0]

        row = compare_pair(str(pred_path), str(gt_path))

        if args.plot_histograms:
            pred_arr = np.clip(load_volume_hu(str(pred_path)), HU_MIN, HU_MAX)
            gt_arr   = np.clip(load_volume_hu(str(gt_path)),   HU_MIN, HU_MAX)
            hist_out = str(Path(args.output).parent / "histograms" / f"{subj_id}_hu_hist.png")
            plot_hu_histogram(pred_arr, gt_arr, title=subj_id, out_path=hist_out)

        records.append(row)

    if not records:
        print("No valid pairs found.")
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"  Subjects compared : {len(records)}")
    print(f"  {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*40}")
    summary_keys = ["rmse", "mae"] + [f"mae_{t}" for t in TISSUE_RANGES]
    for key in summary_keys:
        vals = [r[key] for r in records if not (isinstance(r[key], float) and np.isnan(r[key]))]
        if vals:
            print(f"  {key:<20} {np.mean(vals):>10.3f} {np.std(vals):>10.3f}")
    print(f"{'='*60}\n")

    import pandas as pd
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"Results saved to: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HU analysis tool (R2-9)")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- data ----
    p_data = sub.add_parser("data", help="원본 데이터셋 HU 분포 분석")
    p_data.add_argument("--data_path", required=True)
    p_data.add_argument("--anatomy", nargs="+", default=["AB", "HN", "TH"])
    p_data.add_argument("--modality", nargs="+", default=["cbct", "ct"])
    p_data.add_argument("--output", default="results/hu_analysis.csv")

    # ---- compare ----
    p_cmp = sub.add_parser("compare", help="예측 CT vs GT CT 비교")
    p_cmp.add_argument("--pred_dir", required=True, help="추론 결과 .mha 폴더 (infer_2d 출력)")
    p_cmp.add_argument("--gt_dir",   required=True, help="GT 데이터셋 루트 (anatomy/subj_id/ct.mha)")
    p_cmp.add_argument("--output",   default="results/hu_compare.csv")
    p_cmp.add_argument("--plot_histograms", action="store_true", help="피험자별 HU 히스토그램 저장")

    args = parser.parse_args()
    {"data": cmd_data, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
