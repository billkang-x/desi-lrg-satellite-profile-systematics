"""Validate the minimal AbacusSummit tree needed for AbacusHOD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, default=Path("data/abacus_summit_minimal"))
    parser.add_argument("--sim-name", default="AbacusSummit_base_c000_ph000")
    parser.add_argument("--zdir", default="z0.800")
    parser.add_argument("--min-gb", type=float, default=80.0)
    parser.add_argument("--output", type=Path, default=Path("results/abacus_minimal_tree_check.json"))
    args = parser.parse_args(argv)

    zroot = args.sim_root / args.sim_name / "halos" / args.zdir
    required = ["halo_info", "halo_rv_A"]
    entries = {}
    total = 0
    ok = True
    for name in required:
        path = zroot / name
        nbytes = size_bytes(path)
        total += nbytes
        entries[name] = {
            "path": str(path),
            "exists": path.exists(),
            "n_files": count_files(path),
            "size_gb": nbytes / 1024**3,
        }
        ok = ok and path.exists() and entries[name]["n_files"] > 0
    total_gb = total / 1024**3
    ok = ok and total_gb >= args.min_gb
    summary = {
        "ok": bool(ok),
        "sim_root": str(args.sim_root),
        "sim_name": args.sim_name,
        "zdir": args.zdir,
        "total_size_gb": total_gb,
        "min_expected_gb": args.min_gb,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
