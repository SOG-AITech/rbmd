#!/usr/bin/env python3
"""Compute partial RDFs from an RBMD trajectory in parallel by frame."""

from __future__ import annotations

import argparse
import csv
import mmap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PAIRS = (("1-1", 1, 1, "g_11"), ("1-2", 1, 2, "g_12"), ("2-2", 2, 2, "g_22"))


def frame_offsets(path: Path) -> list[int]:
    marker = b"ITEM: TIMESTEP\n"
    offsets = []
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            position = 0
            while True:
                position = mapped.find(marker, position)
                if position < 0:
                    break
                offsets.append(position)
                position += len(marker)
    return offsets


def read_frame(handle, offset: int):
    handle.seek(offset)
    if handle.readline().strip() != b"ITEM: TIMESTEP":
        raise ValueError(f"invalid frame marker at offset {offset}")
    step = int(handle.readline())
    if handle.readline().strip() != b"ITEM: NUMBER OF ATOMS":
        raise ValueError(f"invalid atom-count header at step {step}")
    atom_count = int(handle.readline())
    if not handle.readline().startswith(b"ITEM: BOX BOUNDS"):
        raise ValueError(f"invalid box header at step {step}")
    bounds = np.asarray(
        [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
    )
    columns = handle.readline().decode("utf-8").strip().split()[2:]
    required = ("type", "x", "y", "z")
    if any(name not in columns for name in required):
        raise ValueError(f"trajectory lacks {required} at step {step}")
    data = np.loadtxt(handle, max_rows=atom_count, ndmin=2)
    if data.shape[0] != atom_count:
        raise ValueError(f"incomplete frame at step {step}")
    atom_types = data[:, columns.index("type")].astype(np.int32, copy=False)
    positions = data[:, [columns.index("x"), columns.index("y"), columns.index("z")]]
    return step, bounds, atom_types, positions


def compute_chunk(arguments):
    path_text, offsets, edges = arguments
    sums = {label: np.zeros(len(edges) - 1, dtype=np.float64) for label, *_ in PAIRS}
    sampled_steps = []
    with Path(path_text).open("rb") as handle:
        for offset in offsets:
            step, bounds, atom_types, positions = read_frame(handle, offset)
            lengths = bounds[:, 1] - bounds[:, 0]
            volume = float(np.prod(lengths))
            wrapped = np.mod(positions - bounds[:, 0], lengths)
            shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
            trees = {}
            counts = {}
            for atom_type in (1, 2):
                selected = wrapped[atom_types == atom_type]
                counts[atom_type] = len(selected)
                trees[atom_type] = cKDTree(selected, boxsize=lengths)
            for label, lhs, rhs, _ in PAIRS:
                cumulative = trees[lhs].count_neighbors(trees[rhs], edges, cumulative=True)
                shell_counts = np.diff(np.asarray(cumulative, dtype=np.float64))
                if lhs == rhs:
                    ideal = counts[lhs] * max(counts[lhs] - 1, 0) * shell_volumes / volume
                else:
                    ideal = counts[lhs] * counts[rhs] * shell_volumes / volume
                sums[label] += np.divide(
                    shell_counts,
                    ideal,
                    out=np.zeros_like(shell_counts),
                    where=ideal > 0,
                )
            sampled_steps.append(step)
    return sums, sampled_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--frames-per-task", type=int, default=8)
    parser.add_argument("--start-step", type=int, default=10)
    parser.add_argument("--end-step", type=int, default=20000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--threshold-percent", type=float, default=3.0)
    args = parser.parse_args()

    all_offsets = frame_offsets(args.trajectory)
    selected_offsets = []
    with args.trajectory.open("rb") as handle:
        for offset in all_offsets:
            handle.seek(offset + len(b"ITEM: TIMESTEP\n"))
            step = int(handle.readline())
            if args.start_step <= step <= args.end_step and step % args.interval == 0:
                selected_offsets.append(offset)
    expected = (args.end_step - args.start_step) // args.interval + 1
    if len(selected_offsets) != expected:
        raise ValueError(f"expected {expected} frames, found {len(selected_offsets)}")

    edges = np.linspace(0.0, 8.0, 801)
    radii = 0.5 * (edges[:-1] + edges[1:])
    chunks = [
        selected_offsets[index : index + args.frames_per_task]
        for index in range(0, len(selected_offsets), args.frames_per_task)
    ]
    totals = {label: np.zeros(800, dtype=np.float64) for label, *_ in PAIRS}
    sampled_steps = []
    completed_frames = 0
    print(
        f"RDF_INDEX frames={len(selected_offsets)} chunks={len(chunks)} "
        f"workers={args.workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(compute_chunk, (str(args.trajectory), chunk, edges)) for chunk in chunks]
        for future in as_completed(futures):
            sums, steps = future.result()
            for label in totals:
                totals[label] += sums[label]
            sampled_steps.extend(steps)
            completed_frames += len(steps)
            print(f"RDF_PROGRESS {completed_frames}/{len(selected_offsets)}", flush=True)

    means = {label: values / len(sampled_steps) for label, values in totals.items()}
    reference = np.genfromtxt(args.reference, delimiter=",", names=True)
    if len(reference["r"]) != len(radii) or not np.allclose(reference["r"], radii, atol=5e-6, rtol=0):
        raise ValueError("reference and RBMD RDF grids differ")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for label, _, _, reference_column in PAIRS:
        values = means[label]
        ref_values = reference[reference_column]
        peak_mask = (radii >= 0.8) & (radii <= 1.8)
        peak_indices = np.flatnonzero(peak_mask)
        rbmd_peak_index = int(peak_indices[np.argmax(values[peak_mask])])
        ref_peak_index = int(peak_indices[np.argmax(ref_values[peak_mask])])
        peak_difference_percent = (values[rbmd_peak_index] / ref_values[ref_peak_index] - 1.0) * 100.0
        mae_percent = np.mean(np.abs(values - ref_values)) / np.mean(np.abs(ref_values)) * 100.0
        rmse = float(np.sqrt(np.mean((values - ref_values) ** 2)))
        summary_rows.append({
            "pair": label,
            "rbmd_peak_r": radii[rbmd_peak_index],
            "rbmd_peak_g": values[rbmd_peak_index],
            "reference_peak_r": radii[ref_peak_index],
            "reference_peak_g": ref_values[ref_peak_index],
            "peak_difference_percent": peak_difference_percent,
            "full_curve_mae_percent": mae_percent,
            "full_curve_rmse": rmse,
            "threshold_percent": args.threshold_percent,
            "within_threshold": "yes" if abs(peak_difference_percent) <= args.threshold_percent else "no",
        })
        with (args.outdir / f"rdf_{label}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("r", "rbmd_g_r", "reference_g_r", "difference"))
            writer.writerows(zip(radii, values, ref_values, values - ref_values))

    with (args.outdir / "rdf_comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.outdir / "metadata.txt").write_text(
        f"sampled_frames={len(sampled_steps)}\n"
        f"sampled_step_min={min(sampled_steps)}\n"
        f"sampled_step_max={max(sampled_steps)}\n"
        "bins=800\nrmax=8\nparallel_workers=" + str(args.workers) + "\n",
        encoding="utf-8",
    )
    for row in summary_rows:
        print(
            f"{row['pair']} peak_diff={row['peak_difference_percent']:+.6f}% "
            f"mae={row['full_curve_mae_percent']:.6f}% pass={row['within_threshold']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
