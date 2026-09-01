#!/usr/bin/env python3
"""Estimate GPU count for replicated ljsalt RBL runs.

The script uses the existing reader-level replicate feature: each test point is
described by cards and replicate dims.  Passing test points give a measured
atoms-per-card lower bound; OOM/failing points give an upper bound.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASE_DATA = Path("examples/ljsalt/lj_salt_20w_v.data")
DEFAULT_TARGET_ATOMS = 1_000_000_000


@dataclass(frozen=True)
class RunPoint:
    cards: int
    dims: tuple[int, int, int]
    status: str

    @property
    def multiplier(self) -> int:
        return self.dims[0] * self.dims[1] * self.dims[2]


def read_base_atoms(path: Path) -> int:
    atom_re = re.compile(r"^\s*(\d+)\s+atoms\s*$")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            match = atom_re.match(line)
            if match:
                return int(match.group(1))
    raise ValueError(f"cannot find '<N> atoms' in {path}")


def parse_dims(text: str) -> tuple[int, int, int]:
    parts = re.split(r"[x,]", text.strip().lower())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("dims must look like 8x8x8")
    dims = tuple(int(part) for part in parts)
    if any(value <= 0 for value in dims):
        raise argparse.ArgumentTypeError("dims values must be positive")
    return dims  # type: ignore[return-value]


def parse_run_point(text: str) -> RunPoint:
    # Format: cards:dims:status, for example 8:8x8x8:pass
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "run point must look like cards:dims:status, e.g. 8:8x8x8:pass"
        )
    cards = int(parts[0])
    dims = parse_dims(parts[1])
    status = parts[2].strip().lower()
    if cards <= 0:
        raise argparse.ArgumentTypeError("cards must be positive")
    if status not in {"pass", "ok", "oom", "fail"}:
        raise argparse.ArgumentTypeError("status must be pass/ok/oom/fail")
    if status == "ok":
        status = "pass"
    return RunPoint(cards=cards, dims=dims, status=status)


def balanced_dims_for_multiplier(target_multiplier: int) -> tuple[int, int, int]:
    if target_multiplier <= 1:
        return (1, 1, 1)
    limit = max(2, math.ceil(target_multiplier ** (1 / 3)) + 12)
    best: tuple[float, tuple[int, int, int]] | None = None
    for x in range(1, limit + 1):
        for y in range(x, limit + 1):
            z = max(y, round(target_multiplier / (x * y)))
            for candidate_z in {max(y, z - 1), max(y, z), max(y, z + 1)}:
                product = x * y * candidate_z
                rel_error = abs(product - target_multiplier) / target_multiplier
                balance = candidate_z / x
                score = rel_error * 100.0 + balance * 0.01
                item = (score, (x, y, candidate_z))
                if best is None or item < best:
                    best = item
    assert best is not None
    return best[1]


def print_plan(base_atoms: int) -> None:
    print("Recommended 4/8-card replicate test points")
    print("base_atoms =", base_atoms)
    print()
    print("cards atoms/card-target replicate total_atoms atoms/card")
    for cards in (4, 8):
        for atoms_per_card_target in (10_000_000, 12_000_000, 14_000_000, 16_000_000):
            multiplier = round(cards * atoms_per_card_target / base_atoms)
            dims = balanced_dims_for_multiplier(multiplier)
            total_atoms = base_atoms * dims[0] * dims[1] * dims[2]
            atoms_per_card = total_atoms / cards
            print(
                f"{cards:5d} {atoms_per_card_target:17,d} "
                f"{dims[0]}x{dims[1]}x{dims[2]:<7} "
                f"{total_atoms:11,d} {atoms_per_card:14,.0f}"
            )


def print_estimate(
    base_atoms: int,
    target_atoms: int,
    safety: float,
    run_points: list[RunPoint],
) -> None:
    if not run_points:
        return

    pass_caps: list[float] = []
    fail_caps: list[float] = []

    print()
    print("Measured points")
    print("cards replicate status total_atoms atoms/card")
    for point in run_points:
        total_atoms = base_atoms * point.multiplier
        atoms_per_card = total_atoms / point.cards
        print(
            f"{point.cards:5d} "
            f"{point.dims[0]}x{point.dims[1]}x{point.dims[2]:<9} "
            f"{point.status:6s} {total_atoms:11,d} {atoms_per_card:14,.0f}"
        )
        if point.status == "pass":
            pass_caps.append(atoms_per_card)
        else:
            fail_caps.append(atoms_per_card)

    print()
    if pass_caps:
        measured_cap = max(pass_caps)
        conservative_cap = measured_cap * safety
        cards = math.ceil(target_atoms / conservative_cap)
        print(f"best_pass_atoms_per_card = {measured_cap:,.0f}")
        print(f"safety = {safety:.2f}")
        print(f"conservative_atoms_per_card = {conservative_cap:,.0f}")
        print(f"estimated_cards_for_{target_atoms:,}_atoms = {cards}")
    else:
        print("No passing point was provided; cannot make a conservative estimate.")

    if fail_caps:
        first_fail = min(fail_caps)
        optimistic_lower_cards = math.floor(target_atoms / first_fail) + 1
        print(f"first_fail_atoms_per_card = {first_fail:,.0f}")
        print(f"cards_must_be_greater_than_about = {optimistic_lower_cards}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan 4/8-card ljsalt replicate tests and estimate cards for 1B atoms."
    )
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--target-atoms", type=int, default=DEFAULT_TARGET_ATOMS)
    parser.add_argument(
        "--safety",
        type=float,
        default=0.85,
        help="fraction of measured max atoms/card to use for production estimate",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        type=parse_run_point,
        help="test result as cards:dims:status, e.g. --run 8:8x8x8:pass",
    )
    args = parser.parse_args()

    if not (0 < args.safety <= 1):
        raise ValueError("--safety must be in (0, 1]")

    base_atoms = read_base_atoms(args.base_data)
    print_plan(base_atoms)
    print_estimate(base_atoms, args.target_atoms, args.safety, args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
