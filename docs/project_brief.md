# Project brief

## Working title

Constraining satellite fraction and velocity bias from DESI DR1 galaxy clustering with HOD/SHAM mocks

## Core question

Can DESI DR1 LRG redshift-space clustering constrain the satellite fraction and central/satellite velocity bias of the galaxy-halo connection strongly enough to improve mock catalogs and small-scale RSD modeling?

This is deliberately narrower than a generic HOD fit. The paper should not merely reproduce DESI clustering. The sellable result is a set of constraints on:

- `f_sat`: satellite fraction.
- `alpha_c`: central velocity bias.
- `alpha_s`: satellite velocity bias.
- Secondary derived quantities: mean host halo mass, large-scale galaxy bias, and the degeneracy between `f_sat` and velocity bias.

## Baseline sample

Start with DESI DR1 LRG clustering catalogs, version `v1.5`.

Primary sample:

- Tracer: `LRG`.
- Caps: `NGC` and `SGC`, fit jointly after separate smoke tests.
- Initial redshift range: `0.4 < z < 1.1`.
- Tomographic splits to test after full-range validation: `0.4-0.6`, `0.6-0.8`, `0.8-1.1`.

Reasoning:

- LRG HOD is better behaved than ELG HOD.
- LRG clustering has high signal-to-noise and a comparatively mature DESI analysis context.
- DESI DR1 public LSS catalogs include clustering-ready data, randoms, `n(z)`, and weights.

## Measurements

Minimum viable data vector:

- Projected clustering `wp(rp)` on small and intermediate scales, e.g. `0.3-30 Mpc/h`.
- Redshift-space multipoles `xi_0(s)` and `xi_2(s)`, e.g. `2-60 Mpc/h`.
- Number density `n(z)` or integrated number density as a likelihood term.

Interpretation:

- `wp(rp)` anchors the one-halo term and therefore `f_sat`.
- `xi_2(s)` and small-scale anisotropy help constrain velocity bias.
- `xi_0(s)` keeps the overall clustering amplitude and halo-mass scale under control.

Avoid in the first pass:

- BAO-scale systematics as a primary result.
- Full cosmological inference.
- ELG selection modeling.
- Assembly bias as a headline parameter. Keep it as a robustness extension.

## Modeling route

Baseline model:

- AbacusSummit boxes with AbacusHOD.
- Zheng-style LRG HOD with parameters such as `logM_cut`, `logM_1`, `sigma`, `alpha`, `kappa`.
- Velocity-bias extension with `alpha_c` and `alpha_s`.

Optional extensions:

- SHAM/abundance matching as a comparison model after the baseline HOD chain is stable.
- Assembly bias parameters only after the no-assembly-bias model gives a clean baseline.
- Survey-window realism through DESI DR1 Abacus mocks or approximate lightcone/window tests.

## Covariance

Practical first stage:

- Use jackknife or a reduced mock set only for code validation.

Publication stage:

- Use DESI DR1 public EZmocks for covariance if their exact LRG statistic and binning can be matched.
- Use Abacus mocks for model validation and bias tests, not as the only covariance source.

## First three milestones

1. Data smoke test:
   - Confirm exact LRG DR1 v1.5 file paths.
   - Read FITS columns.
   - Count objects in the baseline redshift range.
   - Plot `n(z)` for NGC and SGC.

2. Measurement validation:
   - Measure `wp(rp)` and `xi_l(s)` for a small random subset.
   - Reproduce the qualitative scale and shape of DESI public clustering products.
   - Lock binning and weights.

3. Modeling validation:
   - Run AbacusHOD on one simulation redshift near the LRG effective redshift.
   - Fit a reduced data vector.
   - Recover posterior constraints on `f_sat`, `alpha_c`, and `alpha_s`.

## Likely paper figures

1. Redshift distribution and sky split sanity checks.
2. Measured `wp(rp)`, `xi_0`, and `xi_2` for DESI DR1 LRG.
3. Best-fit HOD/SHAM mock comparison to data.
4. Posterior corner plot for HOD and velocity-bias parameters.
5. Derived `f_sat` and host-halo mass distribution.
6. Degeneracy plot showing how `f_sat`, `alpha_c`, and `alpha_s` move small-scale RSD.
7. Robustness checks: cap split, redshift split, scale cuts, optional assembly-bias switch.

## Main risks

- DESI official analyses already cover large-scale clustering, so the paper must emphasize galaxy-halo physics rather than cosmological parameters.
- Velocity bias can be degenerate with satellite fraction and satellite radial profile.
- AbacusHOD setup can be I/O-heavy; start with one box and one redshift before scaling.
- Exact DESI weights and random handling need to be matched carefully before interpreting small-scale differences.

