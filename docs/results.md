# Benchmark Results

## Status

Placeholder — fill in after running benchmark scripts.

To populate this table, run:

    python scripts/generate_mock.py --n 1000 --synthetic
    python scripts/run_baseline.py  --n 1000 --backend scipy
    python scripts/run_rg.py        --n 1000 --backend scipy
    python scripts/compare.py       --n 1000 --plot

## Accuracy: |logL_RG - logL_MLF|

| N    | max |ΔlogL| | Δ fsigma8 | Notes          |
|------|------------|-----------|----------------|
| 100  | —          | —         | dense C        |
| 1000 | —          | —         | dense C        |
| 5000 | —          | —         | dense C        |

Expected: max |ΔlogL| < 1×10⁻⁶ for dense covariance (floating-point only).

## Timing: wall time per logL evaluation

| N    | MLF (s) | RG (s) | Speedup | Backend |
|------|---------|--------|---------|---------|
| 100  | —       | —      | —       | scipy   |
| 1000 | —       | —      | —       | scipy   |
| 5000 | —       | —      | —       | scipy   |
| 1000 | —       | —      | —       | GPU     |
| 5000 | —       | —      | —       | GPU     |

## Best-fit fsigma8

| Method | N    | fsigma8_best | σ(fsigma8) | Notes |
|--------|------|--------------|------------|-------|
| MLF    | 1000 | —            | —          | —     |
| RG     | 1000 | —            | —          | —     |

True value (AbacusSummit c000): fsigma8 = f × σ_8 ≈ 0.47
(f ≈ Ω_m^0.55 ≈ 0.58 at z~0, σ_8 = 0.811)
