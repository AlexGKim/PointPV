#!/usr/bin/env python
"""
Compare MLF baseline and RG method results.

Loads result files produced by run_baseline.py and run_rg.py, then
prints an accuracy and timing comparison.

Usage
-----
    python scripts/compare.py --n 1000
    python scripts/compare.py --baseline results/baseline_1000.npz \
                               --rg results/rg_1000.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MLF and RG results")
    p.add_argument("--n", type=int, default=1000,
                   help="Catalog size (used to find default result files)")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline .npz.  Default: results/baseline_{n}.npz")
    p.add_argument("--rg", default=None,
                   help="Path to RG .npz.  Default: results/rg_{n}.npz")
    p.add_argument("--plot", action="store_true",
                   help="Generate comparison plot (requires matplotlib)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline_path = args.baseline or f"results/baseline_{args.n}.npz"
    rg_path = args.rg or f"results/rg_{args.n}.npz"

    for path in (baseline_path, rg_path):
        if not Path(path).exists():
            print(f"ERROR: result file not found: {path}\n"
                  "Run run_baseline.py and run_rg.py first.",
                  file=sys.stderr)
            sys.exit(1)

    baseline = np.load(baseline_path)
    rg_res = np.load(rg_path)

    from pointpv.benchmark.accuracy import compare_logL, print_accuracy_report, plot_comparison
    from pointpv.benchmark.timing import print_timing_table

    fsigma8 = baseline["fsigma8"]
    logL_mlf = baseline["logL"]
    logL_rg = rg_res["logL"]
    t_mlf = baseline["time_per_eval"]
    t_rg = rg_res["time_per_eval"]

    # Accuracy comparison
    acc = compare_logL(fsigma8, logL_mlf, logL_rg)
    print_accuracy_report(acc)

    # Timing comparison (print_timing_table expects N/mean_time/std_time/logL)
    n_obj = int(baseline["n"]) if "n" in baseline else args.n
    timing_mlf = {
        "N": np.array([n_obj]),
        "mean_time": np.array([float(t_mlf.mean())]),
        "std_time": np.array([float(t_mlf.std())]),
        "logL": np.array([float(logL_mlf.max())]),
    }
    timing_rg = {
        "N": np.array([n_obj]),
        "mean_time": np.array([float(t_rg.mean())]),
        "std_time": np.array([float(t_rg.std())]),
        "logL": np.array([float(logL_rg.max())]),
    }
    print_timing_table(timing_mlf, label="MLF baseline")
    print_timing_table(timing_rg, label="RG method")

    if args.plot:
        out_png = f"results/compare_{args.n}.png"
        plot_comparison(fsigma8, logL_mlf, logL_rg, output_path=out_png)
        print(f"Plot saved to {out_png}")


if __name__ == "__main__":
    main()
