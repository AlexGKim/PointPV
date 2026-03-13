"""
Accuracy comparison utilities for MLF vs. McDonald RG likelihood.

Compares log-likelihood values and derived fsigma8 estimates between
the two methods.  For exact Gaussian data, the RG and Cholesky log-likelihoods
should agree to better than 10⁻¹⁰ absolute (limited only by floating-point).
A sparsity cutoff on the covariance will introduce a small systematic error.

Inputs
------
logL_mlf, logL_rg : (M,) ndarray
    Log-likelihood grids from the two methods, evaluated at the same fsigma8
    grid points.

Outputs
-------
dict
    Comparison statistics: max absolute difference, relative difference,
    best-fit fsigma8 from each method.
"""

from __future__ import annotations

import numpy as np


def compare_logL(
    fsigma8: np.ndarray,
    logL_mlf: np.ndarray,
    logL_rg: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Compare log-likelihood grids from MLF and RG methods.

    Parameters
    ----------
    fsigma8 : (M,) array
        Grid of fsigma8 values.
    logL_mlf : (M,) array
        Log-likelihood from baseline MLF (Cholesky).
    logL_rg : (M,) array
        Log-likelihood from McDonald RG method.

    Returns
    -------
    dict with keys:
        'delta_logL'        : (M,) array of logL_rg - logL_mlf
        'max_abs_diff'      : max |logL_rg - logL_mlf|
        'max_rel_diff'      : max |logL_rg - logL_mlf| / max |logL_mlf|
        'fsigma8_mlf'       : best-fit fsigma8 from MLF
        'fsigma8_rg'        : best-fit fsigma8 from RG
        'delta_fsigma8'     : fsigma8_rg - fsigma8_mlf
    """
    delta = logL_rg - logL_mlf
    max_abs = float(np.max(np.abs(delta)))
    max_rel = max_abs / max(float(np.max(np.abs(logL_mlf))), 1e-30)

    fs8_mlf = float(fsigma8[np.argmax(logL_mlf)])
    fs8_rg = float(fsigma8[np.argmax(logL_rg)])

    return {
        "delta_logL": delta,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "fsigma8_mlf": fs8_mlf,
        "fsigma8_rg": fs8_rg,
        "delta_fsigma8": fs8_rg - fs8_mlf,
    }


def print_accuracy_report(comparison: dict) -> None:
    """Print a human-readable accuracy comparison report."""
    print("\n=== Accuracy Comparison: MLF vs. RG ===")
    print(f"  Max |ΔlogL|:          {comparison['max_abs_diff']:.2e}")
    print(f"  Max rel |ΔlogL|:      {comparison['max_rel_diff']:.2e}")
    print(f"  Best-fit fsigma8 (MLF): {comparison['fsigma8_mlf']:.4f}")
    print(f"  Best-fit fsigma8 (RG):  {comparison['fsigma8_rg']:.4f}")
    print(f"  Δ fsigma8:              {comparison['delta_fsigma8']:.4f}")


def plot_comparison(
    fsigma8: np.ndarray,
    logL_mlf: np.ndarray,
    logL_rg: np.ndarray,
    output_path: str | None = None,
) -> None:
    """
    Two-panel plot: log-likelihood curves and their difference.

    Parameters
    ----------
    fsigma8 : (M,) array
    logL_mlf : (M,) array
    logL_rg : (M,) array
    output_path : str, optional
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # Normalise by max for visual comparison
    logL_mlf_norm = logL_mlf - np.max(logL_mlf)
    logL_rg_norm = logL_rg - np.max(logL_rg)

    ax1.plot(fsigma8, logL_mlf_norm, label="MLF (Cholesky)", color="#1f77b4")
    ax1.plot(fsigma8, logL_rg_norm, label="McDonald RG", color="#ff7f0e",
             linestyle="--", dashes=(6, 2))
    ax1.set_ylabel(r"$\Delta\log L$")
    ax1.set_title(r"$\log L$ vs. $f\sigma_8$")
    ax1.legend()
    ax1.axhline(-0.5, color="gray", linestyle=":", linewidth=0.8,
                label=r"$-\frac{1}{2}\sigma$ level")

    delta = logL_rg - logL_mlf
    ax2.plot(fsigma8, delta, color="#2ca02c")
    ax2.axhline(0, color="gray", linestyle=":")
    ax2.set_xlabel(r"$f\sigma_8$")
    ax2.set_ylabel(r"$\log L_{\rm RG} - \log L_{\rm MLF}$")
    ax2.set_title("Log-likelihood Difference")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()
