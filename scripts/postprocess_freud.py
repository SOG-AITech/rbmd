#!/usr/bin/env python3
"""Post-process RBMD LAMMPS-style trajectories with freud and NumPy.

The script is intentionally standalone: the simulator only writes a trajectory
and invokes this file on rank 0 after the trajectory is closed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import freud
except ImportError as exc:  # pragma: no cover - depends on user environment.
    raise SystemExit(
        "freud is required for analysis postprocess. Install it in the Python "
        "environment used by outputs.analysis_postprocess.python."
    ) from exc


@dataclass
class Frame:
    timestep: int
    box: tuple[float, float, float, float, float, float]
    ids: np.ndarray
    types: np.ndarray
    positions: np.ndarray
    unwrapped: np.ndarray
    velocities: np.ndarray


@dataclass
class RdfPair:
    label_lhs: str
    label_rhs: str
    lhs_types: set[int]
    rhs_types: set[int]


def strip_json_comments(text: str) -> str:
    """Remove C/C++ style comments without touching quoted strings."""
    if text.startswith("\ufeff"):
        text = text[1:]

    result: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(strip_json_comments(path.read_text(encoding="utf-8")))


def resolve_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def read_lammpstrj(path: Path) -> list[Frame]:
    frames: list[Frame] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                continue

            timestep = int(handle.readline().strip())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"{path}: invalid trajectory near timestep {timestep}")
            num_atoms = int(handle.readline().strip())

            bounds_header = handle.readline().strip()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{path}: missing BOX BOUNDS at timestep {timestep}")
            xlo, xhi = map(float, handle.readline().split()[:2])
            ylo, yhi = map(float, handle.readline().split()[:2])
            zlo, zhi = map(float, handle.readline().split()[:2])

            atoms_header = handle.readline().strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"{path}: missing ATOMS header at timestep {timestep}")
            columns = atoms_header.split()[2:]
            col = {name: idx for idx, name in enumerate(columns)}
            required = {"id", "type", "x", "y", "z", "vx", "vy", "vz"}
            missing = required - set(col)
            if missing:
                raise ValueError(
                    f"{path}: missing columns {sorted(missing)} at timestep {timestep}"
                )

            rows = [handle.readline().split() for _ in range(num_atoms)]
            ids = np.array([int(row[col["id"]]) for row in rows], dtype=np.int64)
            order = np.argsort(ids)
            ids = ids[order]
            types = np.array([int(row[col["type"]]) for row in rows], dtype=np.int64)[
                order
            ]

            positions = np.array(
                [[float(row[col["x"]]), float(row[col["y"]]), float(row[col["z"]])] for row in rows],
                dtype=np.float32,
            )[order]
            if {"xu", "yu", "zu"}.issubset(col):
                unwrapped = np.array(
                    [
                        [
                            float(row[col["xu"]]),
                            float(row[col["yu"]]),
                            float(row[col["zu"]]),
                        ]
                        for row in rows
                    ],
                    dtype=np.float32,
                )[order]
            else:
                print(
                    "warning: trajectory has no xu/yu/zu columns; MSD will use wrapped x/y/z",
                    file=sys.stderr,
                )
                unwrapped = positions.copy()

            velocities = np.array(
                [
                    [float(row[col["vx"]]), float(row[col["vy"]]), float(row[col["vz"]])]
                    for row in rows
                ],
                dtype=np.float32,
            )[order]

            frames.append(
                Frame(
                    timestep=timestep,
                    box=(xlo, xhi, ylo, yhi, zlo, zhi),
                    ids=ids,
                    types=types,
                    positions=positions,
                    unwrapped=unwrapped,
                    velocities=velocities,
                )
            )
    return frames


def selected_frames(
    frames: list[Frame], interval: int, start_step: int | None = None, end_step: int | None = None
) -> list[Frame]:
    if interval <= 0:
        return []
    selected: list[Frame] = []
    for frame in frames:
        if frame.timestep % interval != 0:
            continue
        if start_step is not None and frame.timestep < start_step:
            continue
        if end_step is not None and frame.timestep > end_step:
            continue
        selected.append(frame)
    return selected


def parse_one_based_type(value: int) -> int:
    if value < 1:
        raise ValueError(
            f"analysis atom types are 1-based; got {value}. "
            "Use the same atom type numbering as the trajectory."
        )
    return value


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", label)


def build_rdf_pairs(rdf_config: dict[str, Any]) -> list[RdfPair]:
    atoms_pair = rdf_config.get("atoms_pair") or []
    pairs: list[RdfPair] = []
    for pair in atoms_pair:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        lhs = parse_one_based_type(int(pair[0]))
        rhs = parse_one_based_type(int(pair[1]))
        pairs.append(RdfPair(str(lhs), str(rhs), {lhs}, {rhs}))

    groups = {}
    for group in rdf_config.get("groups") or []:
        if not isinstance(group, dict) or "name" not in group:
            continue
        raw_types = [int(value) for value in group.get("types", [])]
        groups[str(group["name"])] = {parse_one_based_type(value) for value in raw_types}

    for lhs, rhs, *_ in rdf_config.get("group_pairs") or []:
        lhs = str(lhs)
        rhs = str(rhs)
        if lhs in groups and rhs in groups:
            pairs.append(RdfPair(lhs, rhs, groups[lhs], groups[rhs]))

    return pairs


def freud_box_and_shift(frame: Frame) -> tuple[freud.box.Box, np.ndarray]:
    xlo, xhi, ylo, yhi, zlo, zhi = frame.box
    lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=np.float32)
    center = np.array([(xlo + xhi) * 0.5, (ylo + yhi) * 0.5, (zlo + zhi) * 0.5])
    return freud.box.Box(float(lengths[0]), float(lengths[1]), float(lengths[2])), center


def write_rdf(frames: list[Frame], config: dict[str, Any], outdir: Path) -> None:
    rdf_config = config.get("outputs", {}).get("rdf_out")
    if not isinstance(rdf_config, dict):
        return

    interval = int(rdf_config.get("interval", 0))
    radius = float(rdf_config.get("radius", 0.0))
    dr = float(rdf_config.get("dr", 0.0))
    if interval <= 0 or radius <= 0.0 or dr <= 0.0:
        return

    rdf_frames = selected_frames(frames, interval)
    if not rdf_frames:
        print("warning: no trajectory frames selected for RDF", file=sys.stderr)
        return

    pairs = build_rdf_pairs(rdf_config)
    if not pairs:
        print("warning: rdf_out has no valid pairs", file=sys.stderr)
        return

    bins = max(1, int(math.ceil(radius / dr)))
    for pair in pairs:
        rdf = freud.density.RDF(bins=bins, r_max=radius)
        sampled = 0
        for frame in rdf_frames:
            box, center = freud_box_and_shift(frame)
            lhs_mask = np.isin(frame.types, list(pair.lhs_types))
            rhs_mask = np.isin(frame.types, list(pair.rhs_types))
            lhs_positions = frame.positions[lhs_mask] - center
            rhs_positions = frame.positions[rhs_mask] - center
            if len(lhs_positions) == 0 or len(rhs_positions) == 0:
                continue

            if pair.lhs_types == pair.rhs_types:
                if len(rhs_positions) < 2:
                    continue
                rdf.compute(
                    (box, rhs_positions),
                    neighbors={"r_max": radius, "exclude_ii": True},
                    reset=False,
                )
            else:
                rdf.compute(
                    (box, rhs_positions),
                    query_points=lhs_positions,
                    neighbors={"r_max": radius, "exclude_ii": False},
                    reset=False,
                )
            sampled += 1

        path = outdir / f"rdf_{sanitize_label(pair.label_lhs)}_{sanitize_label(pair.label_rhs)}.txt"
        counts = getattr(rdf, "bin_counts", np.zeros_like(rdf.rdf))
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"# lhs_label {pair.label_lhs}\n")
            handle.write(f"# rhs_label {pair.label_rhs}\n")
            handle.write(f"# radius {radius}\n")
            handle.write(f"# dr {dr}\n")
            handle.write(f"# sampled_frames {sampled}\n")
            handle.write("# columns: r g_r counts\n")
            for r_value, g_value, count in zip(rdf.bin_centers, rdf.rdf, counts):
                handle.write(f"{r_value:.10g} {g_value:.10g} {float(count):.10g}\n")


def check_constant_ids(frames: list[Frame]) -> np.ndarray:
    if not frames:
        return np.array([], dtype=np.int64)
    reference = frames[0].ids
    for frame in frames[1:]:
        if len(frame.ids) != len(reference) or not np.array_equal(frame.ids, reference):
            raise ValueError("MSD/VACF require constant atom ids in every selected frame")
    return reference


def write_msd(frames: list[Frame], config: dict[str, Any], outdir: Path) -> None:
    msd_config = config.get("outputs", {}).get("msd_out")
    if not isinstance(msd_config, dict):
        return

    interval = int(msd_config.get("interval", 0))
    start_step = int(msd_config.get("start_step", 0))
    end_step = int(msd_config.get("end_step", 2**63 - 1))
    mode = str(msd_config.get("mode", "direct"))
    msd_frames = selected_frames(frames, interval, start_step, end_step)
    if not msd_frames:
        return

    check_constant_ids(msd_frames)
    positions = np.stack([frame.unwrapped for frame in msd_frames], axis=0)
    msd = freud.msd.MSD(mode=mode)
    msd.compute(positions)

    timestep = float(config.get("execution", {}).get("timestep", 1.0))
    reference_step = msd_frames[0].timestep
    with (outdir / "msd.txt").open("w", encoding="utf-8") as handle:
        handle.write("# step delta_time msd shared_atoms reference_step\n")
        for frame, msd_value in zip(msd_frames, msd.msd):
            delta_time = (frame.timestep - reference_step) * timestep
            handle.write(
                f"{frame.timestep} {delta_time:.10g} {msd_value:.10g} "
                f"{len(frame.ids)} {reference_step}\n"
            )


def write_vacf(frames: list[Frame], config: dict[str, Any], outdir: Path) -> None:
    vacf_config = config.get("outputs", {}).get("vacf_out")
    if not isinstance(vacf_config, dict):
        return

    interval = int(vacf_config.get("interval", 0))
    start_step = int(vacf_config.get("start_step", 0))
    end_step = int(vacf_config.get("end_step", 2**63 - 1))
    vacf_frames = selected_frames(frames, interval, start_step, end_step)
    if not vacf_frames:
        return

    check_constant_ids(vacf_frames)
    ref_velocities = vacf_frames[0].velocities
    timestep = float(config.get("execution", {}).get("timestep", 1.0))
    reference_step = vacf_frames[0].timestep

    with (outdir / "vacf.txt").open("w", encoding="utf-8") as handle:
        handle.write("# step delta_time vacf shared_atoms reference_step\n")
        for frame in vacf_frames:
            vacf = np.einsum("ij,ij->i", frame.velocities, ref_velocities).mean()
            delta_time = (frame.timestep - reference_step) * timestep
            handle.write(
                f"{frame.timestep} {delta_time:.10g} {vacf:.10g} "
                f"{len(frame.ids)} {reference_step}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process RBMD trajectory outputs with freud."
    )
    parser.add_argument("--config", required=True, help="RBMD JSON config path.")
    parser.add_argument("--traj", default="rbmd.trj", help="LAMMPS-style trajectory path.")
    parser.add_argument("--outdir", default=".", help="Directory for analysis txt outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    traj_path = resolve_path(args.traj, Path.cwd())
    outdir = resolve_path(args.outdir, Path.cwd())
    outdir.mkdir(parents=True, exist_ok=True)

    if not traj_path.exists():
        raise FileNotFoundError(f"trajectory not found: {traj_path}")

    frames = read_lammpstrj(traj_path)
    if not frames:
        raise ValueError(f"trajectory contains no frames: {traj_path}")

    write_rdf(frames, config, outdir)
    write_msd(frames, config, outdir)
    write_vacf(frames, config, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
