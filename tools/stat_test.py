"""
tools/stat_test.py — R1-3 통계적 유의성 검정 (paired t-test / Wilcoxon)

Usage:
    python tools/stat_test.py --results_2d results/baseline_2d.csv --results_2_5d results/baseline_2_5d.csv
    python tools/stat_test.py --results_2d results/baseline_2d.csv --results_2_5d results/baseline_2_5d.csv --test wilcoxon
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["psnr", "ssim", "rmse", "fid"]


def load_metric_col(csv_path: str, metric: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in {csv_path}. Available: {list(df.columns)}")
    return df[metric].dropna().to_numpy(dtype=float)


def run_test(a: np.ndarray, b: np.ndarray, test: str) -> tuple[float, float]:
    """Return (statistic, p_value). a=2D scores, b=2.5D scores."""
    if len(a) != len(b):
        raise ValueError(
            f"Sample sizes differ ({len(a)} vs {len(b)}). "
            "Ensure both CSVs have equal rows (same number of seeds / test samples)."
        )
    if test == "ttest":
        stat, p = stats.ttest_rel(a, b)
    elif test == "wilcoxon":
        stat, p = stats.wilcoxon(a, b)
    else:
        raise ValueError(f"Unknown test: {test}")
    return float(stat), float(p)


def main():
    parser = argparse.ArgumentParser(description="Statistical significance test for R1-3")
    parser.add_argument("--results_2d",   required=True, help="CSV from 2D model runs")
    parser.add_argument("--results_2_5d", required=True, help="CSV from 2.5D model runs")
    parser.add_argument(
        "--test",
        choices=["ttest", "wilcoxon"],
        default="ttest",
        help="Statistical test to use (default: ttest / paired t-test)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
        help=f"Metrics to test (default: {METRICS})",
    )
    parser.add_argument(
        "--output",
        default="results/stat_test_results.csv",
        help="Output CSV path (default: results/stat_test_results.csv)",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    args = parser.parse_args()

    rows = []
    print(f"\n{'='*65}")
    print(f"  Statistical test : {args.test}")
    print(f"  Significance α   : {args.alpha}")
    print(f"  2D results       : {args.results_2d}")
    print(f"  2.5D results     : {args.results_2_5d}")
    print(f"{'='*65}")
    print(f"  {'Metric':<8} {'2D mean':>10} {'2.5D mean':>10} {'stat':>10} {'p-value':>10} {'sig':>5}")
    print(f"  {'-'*58}")

    for metric in args.metrics:
        try:
            scores_2d   = load_metric_col(args.results_2d,   metric)
            scores_2_5d = load_metric_col(args.results_2_5d, metric)
        except (ValueError, KeyError) as e:
            print(f"  [SKIP] {metric}: {e}")
            continue

        stat, p = run_test(scores_2d, scores_2_5d, args.test)
        sig = "*" if p < args.alpha else ""

        print(f"  {metric:<8} {scores_2d.mean():>10.4f} {scores_2_5d.mean():>10.4f} "
              f"{stat:>10.4f} {p:>10.4f} {sig:>5}")

        rows.append({
            "metric": metric,
            "mean_2d":   float(scores_2d.mean()),
            "std_2d":    float(scores_2d.std()),
            "mean_2_5d": float(scores_2_5d.mean()),
            "std_2_5d":  float(scores_2_5d.std()),
            "statistic": stat,
            "p_value":   p,
            "significant": p < args.alpha,
            "test": args.test,
            "n_2d":   len(scores_2d),
            "n_2_5d": len(scores_2_5d),
        })

    print(f"{'='*65}\n")

    if rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
