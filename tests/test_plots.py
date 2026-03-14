"""
Smoke tests for all existing plot-generation functions.

Uses matplotlib Agg backend (no display required) and pytest's tmp_path fixture.
Each test asserts only that the output file exists and is non-empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# benchmark.accuracy.plot_comparison
# ---------------------------------------------------------------------------

def test_plot_comparison(tmp_path: Path) -> None:
    from pointpv.benchmark.accuracy import plot_comparison

    fs8 = np.linspace(0.2, 0.8, 20)
    logL_mlf = -(fs8 - 0.47) ** 2 * 50
    logL_rg = logL_mlf + np.random.default_rng(0).normal(0, 1e-7, len(fs8))

    out = str(tmp_path / "compare.png")
    plot_comparison(fs8, logL_mlf, logL_rg, output_path=out)

    assert Path(out).exists(), "compare.png was not created"
    assert Path(out).stat().st_size > 0, "compare.png is empty"


# ---------------------------------------------------------------------------
# benchmark.timing.plot_scaling
# ---------------------------------------------------------------------------

def test_plot_scaling(tmp_path: Path) -> None:
    from pointpv.benchmark.timing import plot_scaling

    results = {
        "MLF": {
            "N": np.array([50, 100, 200]),
            "mean_time": np.array([0.01, 0.08, 0.6]),
            "std_time": np.array([0.001, 0.005, 0.03]),
            "logL": np.array([-100.0, -200.0, -400.0]),
        },
        "RG": {
            "N": np.array([50, 100, 200]),
            "mean_time": np.array([0.005, 0.015, 0.04]),
            "std_time": np.array([0.0005, 0.001, 0.003]),
            "logL": np.array([-100.0, -200.0, -400.0]),
        },
    }

    out = str(tmp_path / "scaling.png")
    plot_scaling(results, output_path=out)

    assert Path(out).exists(), "scaling.png was not created"
    assert Path(out).stat().st_size > 0, "scaling.png is empty"


# ---------------------------------------------------------------------------
# scripts.plot_catalog.plot_nz and plot_sky
# ---------------------------------------------------------------------------

def test_plot_nz(tmp_path: Path) -> None:
    from plot_catalog import plot_nz

    rng = np.random.default_rng(5)
    z = rng.uniform(0.01, 0.1, 200)

    plot_nz(
        z,
        stem="test",
        outdir=str(tmp_path),
        m_lim=20.0,
        M_star=-21.5,
        alpha=-1.1,
        M_faint=-17.0,
        n_bins=10,
    )

    out = tmp_path / "nz_test.pdf"
    assert out.exists(), "nz_test.pdf was not created"
    assert out.stat().st_size > 0, "nz_test.pdf is empty"


def test_plot_sky(tmp_path: Path) -> None:
    from plot_catalog import plot_sky

    rng = np.random.default_rng(6)
    n = 200
    z = rng.uniform(0.01, 0.1, n)
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))

    plot_sky(z, ra, dec, stem="test", outdir=str(tmp_path))

    out = tmp_path / "sky_test.pdf"
    assert out.exists(), "sky_test.pdf was not created"
    assert out.stat().st_size > 0, "sky_test.pdf is empty"
