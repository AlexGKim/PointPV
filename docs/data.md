# Data: AbacusSummit Light Cone

## Reference

Garrison et al. (2021), MNRAS 508, 575.
AbacusSummit: a massive set of high-accuracy, high-resolution N-body simulations.

## NERSC location

The AbacusSummit light cone files live on Perlmutter.  Confirm the exact path
with your collaborators before use.  A typical path is:

    /global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_base_c000_ph000/

The light cone is stored as ASDF files (one per subslab) and read by
`abacusutils` (`abacus_summit.AbacusSummit`).

## Local development subset

For local development (laptop), copy a small subset of ~2000 halos:

    rsync -avz perlmutter:/path/to/lightcone/L0/ data/lightcone_subset/

Then generate a mock catalog:

    python scripts/generate_mock.py --n 1000 --lightcone data/lightcone_subset/

Or use the synthetic generator (no abacusutils needed):

    python scripts/generate_mock.py --n 1000 --synthetic

## Catalog format (.npz)

| Key        | Shape  | Units   | Description                        |
|------------|--------|---------|------------------------------------|
| ra         | (N,)   | deg     | Right ascension                    |
| dec        | (N,)   | deg     | Declination                        |
| z_obs      | (N,)   | —       | Observed (peculiar+Hubble) redshift |
| eta        | (N,)   | —       | Log-distance ratio η = ln(d_est/d_true) |
| sigma_eta  | (N,)   | —       | Per-object η uncertainty (homoscedastic) |
| v_r_true   | (N,)   | km/s    | True radial peculiar velocity (validation only) |
| pos        | (N,3)  | Mpc/h   | Comoving Cartesian positions       |

## Velocity conversion

The likelihood operates on line-of-sight velocities u (km/s):

    u_i = c × z_i × η_i   (low-z approximation)

implemented in `pointpv/mock/catalog.py:eta_to_velocity`.

## Selection

The mock applies a top-N mass selection from the halo catalog, mimicking
a magnitude-limited survey.  The mass threshold can be set explicitly via
`--mass-cut` in `generate_mock.py`.

## Cosmology

AbacusSummit base cosmology (c000):
- Ω_m = 0.315192
- Ω_b = 0.049352
- H_0 = 67.36 km/s/Mpc
- σ_8 = 0.8111
- n_s = 0.9649

These are fixed in the covariance model; only fsigma8 is varied.
