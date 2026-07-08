# AbacusSummit Minimal Staging Plan

Updated: 2026-06-29

Goal: stage only the minimal AbacusHOD input subset for a DESI LRG `z~0.7`
pilot, avoiding the full multi-TB halo tarball.

## Target

- Simulation: `AbacusSummit_base_c000_ph000`
- Epoch: `z0.800`
- Local ParaCloud target:
  - `/public3/home/scg7816/desi_hod_mocks/data/abacus_summit_minimal/AbacusSummit_base_c000_ph000/halos/z0.800/halo_info`
  - `/public3/home/scg7816/desi_hod_mocks/data/abacus_summit_minimal/AbacusSummit_base_c000_ph000/halos/z0.800/halo_rv_A`

## Expected Size

From the AbacusSummit disk-space table:

- `halos/z0.800/halo_info`: about `76.4 GB`
- `halos/z0.800/halo_rv_A`: about `26.3 GB`
- Minimum raw subset: about `102.7 GB`
- Practical staging budget with temporary files and caches: `200-300 GB`

## Why This Subset

AbacusHOD needs halo properties plus a particle/halo subsample to paint
satellites and velocity bias. The `halo_rv_A` subset is the smallest useful
subsample route for an HOD pilot. We explicitly avoid `halo_rv_B`,
`field_rv_*`, particle PID files, and the full per-simulation halo tarball.

## Data Access Reality

ParaCloud currently has `wget`, `curl`, and `rsync`, but no `globus` or `htar`.
The login node also does not appear to have reliable direct external HTTPS
access to the Abacus portal. Therefore the recommended route is:

1. Use NERSC CFS directly if a NERSC account is available:
   `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit`
2. Or extract the subset from HPSS on NERSC with `htar`.
3. Transfer the extracted subset to ParaCloud with Globus/rsync/scp, depending
   on what endpoints are available.

## Files Prepared

- `configs/abacus_hod_lrg_z0p8_minimal.yaml`
- `scripts/stage_abacus_minimal_from_nersc.sh`
- `scripts/check_abacus_minimal_tree.py`
- `scripts/paracloud_setup_abacushod_env.sh`
- `scripts/slurm_abacushod_export_grid.sh`
