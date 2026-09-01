#!/usr/bin/env python3
"""Compare thermo outputs from a base run and a replicated run.

The replicated system should not have identical extensive energies. This
script compares temperature/pressure directly and compares thermo energies
after normalizing by atom count.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev


@dataclass(frozen=True)
class Series:
    name: str
    values: list[float]


def read_table(path: Path) -> dict[str, Series]:
    if not path.exists():
        return {}

    headers: list[str] | None = None
    columns: dict[str, list[float]] = {}

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if headers is None:
            headers = parts
            columns = {name: [] for name in headers}
            continue

        if len(parts) != len(headers):
            continue

        for name, value in zip(headers, parts):
            columns[name].append(float(value))

    return {name: Series(name, values) for name, values in columns.items()}


def finite_values(series: Series) -> bool:
    return all(math.isfinite(value) for value in series.values)


def is_energy_column(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("e_")
        or "energy" in lowered
        or lowered in {"kspace", "total-potential-energy"}
    )


def tail(values: list[float], count: int) -> list[float]:
    if count <= 0:
        return values
    return values[-count:]


def relative_diff(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale


def compare_series(
    base: Series,
    replicated: Series,
    *,
    base_atoms: int,
    replicated_atoms: int,
    tail_count: int,
    normalize_by_atoms: bool,
) -> dict[str, float | bool | int]:
    base_values = tail(base.values, tail_count)
    replicated_values = tail(replicated.values, tail_count)
    count = min(len(base_values), len(replicated_values))
    base_values = base_values[-count:]
    replicated_values = replicated_values[-count:]

    if normalize_by_atoms:
        base_values = [value / base_atoms for value in base_values]
        replicated_values = [value / replicated_atoms for value in replicated_values]

    diffs = [abs(a - b) for a, b in zip(base_values, replicated_values)]
    rel_diffs = [relative_diff(a, b) for a, b in zip(base_values, replicated_values)]

    return {
        "count": count,
        "base_mean": fmean(base_values) if base_values else math.nan,
        "replicated_mean": fmean(replicated_values) if replicated_values else math.nan,
        "base_std": pstdev(base_values) if len(base_values) > 1 else 0.0,
        "replicated_std": pstdev(replicated_values) if len(replicated_values) > 1 else 0.0,
        "max_abs_diff": max(diffs) if diffs else math.nan,
        "max_rel_diff": max(rel_diffs) if rel_diffs else math.nan,
        "base_finite": finite_values(base),
        "replicated_finite": finite_values(replicated),
    }


def status_for(result: dict[str, float | bool | int], tolerance: float) -> str:
    if not result["base_finite"] or not result["replicated_finite"]:
        return "FAIL"
    if int(result["count"]) == 0:
        return "FAIL"
    if float(result["max_rel_diff"]) > tolerance:
        return "WARN"
    return "PASS"


def print_section(
    title: str,
    base_table: dict[str, Series],
    replicated_table: dict[str, Series],
    *,
    base_atoms: int,
    replicated_atoms: int,
    tail_count: int,
    tolerance: float,
    normalize_energy: bool,
) -> bool:
    common_names = sorted(set(base_table) & set(replicated_table))
    common_names = [name for name in common_names if name.lower() != "step"]

    print(f"\n[{title}]")
    if not common_names:
        print("  FAIL no common numeric columns")
        return False

    all_ok = True
    for name in common_names:
        normalize = normalize_energy and is_energy_column(name)
        result = compare_series(
            base_table[name],
            replicated_table[name],
            base_atoms=base_atoms,
            replicated_atoms=replicated_atoms,
            tail_count=tail_count,
            normalize_by_atoms=normalize,
        )
        status = status_for(result, tolerance)
        if status != "PASS":
            all_ok = False
        unit = "per-atom" if normalize else "raw"
        print(
            "  "
            f"{status:4} {name:<18} {unit:<8} "
            f"n={int(result['count']):<4} "
            f"base_mean={float(result['base_mean']): .8e} "
            f"rep_mean={float(result['replicated_mean']): .8e} "
            f"max_abs={float(result['max_abs_diff']): .8e} "
            f"max_rel={float(result['max_rel_diff']): .8e}"
        )

    missing_base = sorted(set(replicated_table) - set(base_table))
    missing_replicated = sorted(set(base_table) - set(replicated_table))
    if missing_base:
        print(f"  INFO columns only in replicated: {', '.join(missing_base)}")
    if missing_replicated:
        print(f"  INFO columns only in base: {', '.join(missing_replicated)}")

    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RBMD base and replicated thermo outputs."
    )
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--replicated-run", required=True, type=Path)
    parser.add_argument("--base-atoms", required=True, type=int)
    parser.add_argument("--replicated-atoms", required=True, type=int)
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Only compare the last N rows. Default: compare all rows.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Relative-difference threshold for WARN. Default: 0.25.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.base_atoms <= 0 or args.replicated_atoms <= 0:
        raise SystemExit("atom counts must be positive")

    base_temp = read_table(args.base_run / "temperature.txt")
    replicated_temp = read_table(args.replicated_run / "temperature.txt")
    base_thermo = read_table(args.base_run / "thermo.txt")
    replicated_thermo = read_table(args.replicated_run / "thermo.txt")

    ok = True
    ok &= print_section(
        "temperature.txt",
        base_temp,
        replicated_temp,
        base_atoms=args.base_atoms,
        replicated_atoms=args.replicated_atoms,
        tail_count=args.tail,
        tolerance=args.tolerance,
        normalize_energy=False,
    )
    ok &= print_section(
        "thermo.txt",
        base_thermo,
        replicated_thermo,
        base_atoms=args.base_atoms,
        replicated_atoms=args.replicated_atoms,
        tail_count=args.tail,
        tolerance=args.tolerance,
        normalize_energy=True,
    )

    print("\nOverall:", "PASS" if ok else "CHECK_WARNINGS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
