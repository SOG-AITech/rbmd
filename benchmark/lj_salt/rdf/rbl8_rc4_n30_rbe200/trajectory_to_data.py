#!/usr/bin/env python3
"""Convert the last RBMD dump frame into a charge-style LAMMPS data file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def last_frame(path: Path):
    result = None
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                continue
            step = int(handle.readline())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("malformed atom-count header")
            atom_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("malformed box header")
            bounds = np.asarray(
                [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
            )
            columns = handle.readline().strip().split()[2:]
            data = np.loadtxt(handle, max_rows=atom_count, ndmin=2)
            if data.shape[0] != atom_count:
                raise ValueError(f"incomplete frame at step {step}")
            result = step, bounds, columns, data
    if result is None:
        raise ValueError("trajectory contains no frames")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    step, bounds, columns, data = last_frame(args.trajectory)
    required = ("id", "type", "q", "x", "y", "z", "vx", "vy", "vz")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"trajectory lacks columns: {missing}")
    index = {name: columns.index(name) for name in required}
    order = np.argsort(data[:, index["id"]].astype(np.int64))
    data = data[order]

    with args.output.open("w", encoding="utf-8") as out:
        out.write(f"LAMMPS data from RBMD trajectory step {step}\n\n")
        out.write(f"{len(data)} atoms\n2 atom types\n\n")
        for axis, label in enumerate(("x", "y", "z")):
            out.write(f"{bounds[axis, 0]:.17g} {bounds[axis, 1]:.17g} {label}lo {label}hi\n")
        out.write("\nMasses\n\n1 1\n2 1\n")
        out.write("\nPair Coeffs # lj/cut/coul/long\n\n1 1 1\n2 1 1\n")
        out.write("\nAtoms # charge\n\n")
        for row in data:
            out.write(
                f"{int(row[index['id']])} {int(row[index['type']])} "
                f"{row[index['q']]:.17g} {row[index['x']]:.17g} "
                f"{row[index['y']]:.17g} {row[index['z']]:.17g} 0 0 0\n"
            )
        out.write("\nVelocities\n\n")
        for row in data:
            out.write(
                f"{int(row[index['id']])} {row[index['vx']]:.17g} "
                f"{row[index['vy']]:.17g} {row[index['vz']]:.17g}\n"
            )

    print(f"CONVERTED step={step} atoms={len(data)} output={args.output}")


if __name__ == "__main__":
    main()
