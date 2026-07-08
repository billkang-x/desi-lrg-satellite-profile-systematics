# DESI LRG Satellite-Profile Systematics

This repository contains the analysis code, lightweight summary products and
paper source for:

**Constraining satellite fraction and velocity bias from DESI DR1 galaxy
clustering with HOD/SHAM mocks**

Repository URL:
<https://github.com/billkang-x/desi-lrg-satellite-profile-systematics>

The current manuscript presents a diagnostic study of how simplified satellite
selection, satellite radial profile and velocity-bias prescriptions affect
DESI-like LRG clustering constraints. It is not a replacement for a full DESI
AbacusHOD inference.

## Repository contents

- `scripts/`: measurement, model-vector, fitting, scoring and summary scripts.
- `configs/` and `config/`: analysis configuration files.
- `jobs/`: cluster job wrappers used for the Abacus full-box grid tests.
- `results/`: lightweight CSV/JSON/TXT summary products only.
- `docs/`: planning notes and staging documentation.
- `paper/`: LaTeX source, references, figures and the current draft PDF.

The large AbacusSummit halo products, DESI random catalogues, TNG intermediate
files and regenerated binary model vectors are intentionally not tracked. They
live under `data/` in the working copy and are excluded by `.gitignore`.

## Data products not included

The local analysis used public AbacusSummit products, especially compact halo
catalogues from the `AbacusSummit_base_c000_ph000` box at `z=0.8`, plus public
DESI LRG clustering measurements or data vectors derived from them. These inputs
are much larger than is appropriate for GitHub and should be downloaded from the
corresponding public data releases.

Regenerated products with extensions such as `.asdf`, `.fits`, `.hdf5`, `.npy`
and `.npz` are also excluded. The repository keeps the scripts and lightweight
summary tables needed to inspect the workflow and reproduce the paper figures
once the external inputs are staged locally.

## Typical workflow

The exact command choices depend on where the external catalogues are staged,
but the main workflow is:

```bash
python scripts/build_lrg_datavector.py --help
python scripts/measure_lrg_pycorr.py --help
python scripts/build_abacus_fullbox_candidates.py --help
python scripts/fit_abacus_minimal_hod_pycorr.py --help
python scripts/fit_abacus_profile_dilution_pycorr.py --help
python scripts/fit_abacus_nfw_profile_pycorr.py --help
python scripts/summarize_profile_dilution_results.py --help
python scripts/summarize_nfw_profile_results.py --help
```

The manuscript can be built from `paper/main.tex` with a standard LaTeX
installation that provides the MNRAS class and BibTeX support.
