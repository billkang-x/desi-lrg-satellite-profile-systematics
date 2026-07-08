"""Run an AbacusHOD LRG mock and export a compact HDF5 catalog.

The exported file is the input contract used by
``fit_hod_catalog_fsat_alpha.py``: positions, velocities, host halo id,
central/satellite flag, and ``Ncent`` metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_setters(values: list[str]) -> dict[str, float | str]:
    updates: dict[str, float | str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE in --set, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty key in --set {item!r}")
        try:
            updates[key] = float(value)
        except ValueError:
            updates[key] = value
    return updates


def scalar_int(value) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar Ncent, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def require_keys(catalog: dict[str, object], keys: list[str]) -> None:
    missing = [key for key in keys if key not in catalog]
    if missing:
        raise KeyError("AbacusHOD catalog is missing keys: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to abacus_hod.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 catalog")
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--alpha-s", type=float, default=None, help="Override satellite velocity bias")
    parser.add_argument("--alpha-c", type=float, default=None, help="Override central velocity bias")
    parser.add_argument("--set", dest="setters", action="append", default=[], help="Override tracer HOD parameter, e.g. logM_cut=12.8")
    parser.add_argument("--want-rsd", choices=["config", "true", "false"], default="config")
    parser.add_argument("--reseed", type=int, default=None)
    parser.add_argument("--nthread", type=int, default=16)
    args = parser.parse_args(argv)

    import h5py
    import yaml
    from abacusnbody.hod.abacus_hod import AbacusHOD

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    sim_params = config["sim_params"]
    hod_params = config["HOD_params"]
    clustering_params = config.get("clustering_params")

    runner = AbacusHOD(sim_params, hod_params, clustering_params)
    if args.tracer not in runner.tracers:
        raise KeyError(f"Tracer {args.tracer!r} not available. Available: {sorted(runner.tracers)}")

    updates = parse_setters(args.setters)
    if args.alpha_s is not None:
        updates["alpha_s"] = args.alpha_s
    if args.alpha_c is not None:
        updates["alpha_c"] = args.alpha_c
    runner.tracers[args.tracer].update(updates)

    if args.want_rsd == "config":
        want_rsd = bool(hod_params.get("want_rsd", True))
    else:
        want_rsd = args.want_rsd == "true"

    mock = runner.run_hod(
        runner.tracers,
        want_rsd=want_rsd,
        reseed=args.reseed,
        write_to_disk=False,
        Nthread=args.nthread,
    )
    catalog = mock[args.tracer]
    required = ["x", "y", "z", "vx", "vy", "vz", "id", "Ncent"]
    require_keys(catalog, required)
    ncent = scalar_int(catalog["Ncent"])
    ngal = len(np.asarray(catalog["x"]))
    if ncent < 0 or ncent > ngal:
        raise ValueError(f"Invalid Ncent={ncent} for ngal={ngal}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        for key in ["x", "y", "z", "vx", "vy", "vz", "id", "mass"]:
            if key in catalog:
                handle.create_dataset(key, data=np.asarray(catalog[key]), compression="gzip", shuffle=True)
        is_sat = np.arange(ngal, dtype="i8") >= ncent
        handle.create_dataset("is_sat", data=is_sat.astype("i1"), compression="gzip", shuffle=True)
        handle.attrs["Ncent"] = ncent
        handle.attrs["tracer"] = args.tracer
        handle.attrs["want_rsd"] = int(want_rsd)
        handle.attrs["reseed"] = -1 if args.reseed is None else int(args.reseed)
        handle.attrs["hod_updates_json"] = json.dumps(updates, sort_keys=True)
        handle.attrs["source_config"] = str(args.config)

    fsat = float((ngal - ncent) / ngal) if ngal else float("nan")
    print(json.dumps({"output": str(args.output), "ngal": int(ngal), "Ncent": int(ncent), "f_sat": fsat}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
