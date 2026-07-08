# Next measurement plan

## Current status

Completed local smoke tests:

- Downloaded DESI DR1 LRG data catalogs for NGC and SGC.
- Downloaded one public random catalog per cap:
  - `LRG_NGC_0_clustering.ran.fits`
  - `LRG_SGC_0_clustering.ran.fits`
- Verified FITS columns, row counts, weights, and redshift coverage.
- Produced small-subset Landy-Szalay `xi_0` and `xi_2` smoke tests for `0.6 < z < 0.8`.

These smoke tests validate the data flow but are not publication-quality two-point measurements.

## Environment note

The current Windows environment has no free space on `C:`, so packages that build or unpack through default temp paths can fail. In addition, the PyPI package named `pycorr` is not the DESI/cosmodesi correlation-function package; it is an unrelated categorical-correlation package. The DESI/cosmodesi stack should be installed from the cosmodesi GitHub repositories or in a managed conda environment.

Use D-drive cache/temp variables for any local installs:

```powershell
$env:TEMP = 'D:\研究方向\tmp'
$env:TMP = 'D:\研究方向\tmp'
$env:PIP_CACHE_DIR = 'D:\研究方向\pip-cache'
```

## Production two-point stack

Recommended production stack:

- `pycorr` from `github.com/cosmodesi/pycorr`
- `pypower` from `github.com/cosmodesi/pypower`
- `Corrfunc`, if available on the target platform
- `cosmoprimo` for cosmology consistency
- DESI LSS scripts only if we need exact official binning/weights/format matching

Prefer Linux, WSL, or HPC for the production run. This avoids Windows build friction for compiled pair-count backends.

## Production measurement target

Baseline sample:

- DESI DR1 LRG.
- Joint NGC + SGC.
- First production bin: `0.6 < z < 0.8`.
- Then full `0.4 < z < 1.1` and tomographic bins.

Data vector:

- `wp(rp)`, with `rp ~ 0.3-30 Mpc/h` and `pi_max ~ 80 Mpc/h`.
- `xi_0(s)` and `xi_2(s)`, with `s ~ 2-60 Mpc/h`.
- Number density constraint from public `n(z)` files.

Weights:

- Start with DESI catalog `WEIGHT`.
- Validate treatment of `WEIGHT_FKP` separately for configuration-space vs Fourier-space statistics.
- Keep NGC and SGC split diagnostics before combining.

## Immediate next coding tasks

1. Write a production `measure_lrg_pycorr.py` script with:
   - CLI options for cap, z-bin, random index list, and binning.
   - DESI FITS reader.
   - Cartesian coordinate conversion.
   - `TwoPointCorrelationFunction` calls for `smu` and `rppi`.
   - Output to `.npy`, `.csv`, and diagnostic plots.

2. Add a data manifest that records which random catalogs are present.

3. Once a Linux/WSL/HPC environment is available, install the cosmodesi stack and run:
   - SGC-only `0.6 < z < 0.8` smoke test.
   - NGC-only `0.6 < z < 0.8` smoke test.
   - Joint NGC+SGC `0.6 < z < 0.8` measurement.

4. Compare the measured shape and amplitude against DESI public clustering references before starting HOD fitting.

