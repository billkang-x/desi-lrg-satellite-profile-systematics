"""Print pycorr return shapes for the installed ParaCloud version."""

from __future__ import annotations

import numpy as np
from pycorr import TwoPointCorrelationFunction


def describe(name: str, value: object, indent: str = "") -> None:
    if isinstance(value, (tuple, list)):
        print(f"{indent}{name}: {type(value).__name__} len={len(value)}")
        for i, item in enumerate(value):
            describe(f"{name}[{i}]", item, indent + "  ")
    else:
        arr = np.asarray(value)
        print(f"{indent}{name}: {type(value).__name__} shape={arr.shape} dtype={arr.dtype}")


def main() -> int:
    rng = np.random.default_rng(1234)
    boxsize = 100.0
    pos = [rng.uniform(0, boxsize, 1000) for _ in range(3)]
    smu = TwoPointCorrelationFunction(
        "smu",
        edges=(np.linspace(5.0, 30.0, 6), np.linspace(-1.0, 1.0, 9)),
        data_positions1=pos,
        boxsize=boxsize,
        los="z",
        nthreads=2,
        engine="corrfunc",
    )
    describe("smu_call", smu(ells=(0, 2), return_sep=True))
    rppi = TwoPointCorrelationFunction(
        "rppi",
        edges=(np.geomspace(0.5, 30.0, 6), np.linspace(-40.0, 40.0, 9)),
        data_positions1=pos,
        boxsize=boxsize,
        los="z",
        nthreads=2,
        engine="corrfunc",
    )
    describe("rppi_call", rppi(return_sep=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
