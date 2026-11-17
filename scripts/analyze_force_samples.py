#!/usr/bin/env python3
"""Analyze fixed-configuration force CSV shards without gathering forces in RBMD.

RMS is sqrt(mean_atoms(|F-F_ref|^2)), not divided by three components.
Variance is the trace of the per-atom force covariance, averaged over atoms.
Memory is O(N), independent of sample count. Each shard must be sample ordered.
"""
import argparse
import csv
import glob
import heapq
import itertools
import json
import math
from pathlib import Path

if __package__:
    from .merge_trajectory_shards import read_frame
else:
    from merge_trajectory_shards import read_frame


def expand(patterns):
    files = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not files:
        raise ValueError("No input files match")
    return files


def rows(path, id_offset=0):
    previous = -1
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "fx", "fy", "fz"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: requires id,fx,fy,fz columns")
        for row in reader:
            sample = int(row.get("sample", "0"))
            if sample < previous:
                raise ValueError(f"{path}: sample indices are not ordered")
            previous = sample
            force = tuple(float(row[k]) for k in ("fx", "fy", "fz"))
            position = tuple(float(row[k]) for k in ("x", "y", "z")) if {"x", "y", "z"} <= row.keys() else None
            if not all(math.isfinite(v) for v in force + (position or ())):
                raise ValueError(f"{path}: non-finite force or position")
            yield sample, int(row["id"]) + id_offset, force, position


def reference_rows(path, id_offset=0):
    """Accept reference CSVs or native LAMMPS text custom dumps."""
    with open(path) as stream:
        if stream.readline().strip() != "ITEM: TIMESTEP":
            yield from rows(path, id_offset)
            return
        stream.seek(0)
        previous = None
        while True:
            frame = read_frame(stream)
            if frame is None:
                return
            step, _, _, fields, atoms = frame
            if previous is not None and step <= previous:
                raise ValueError(f"{path}: reference timesteps must be strictly increasing")
            previous = step
            required = {"id", "x", "y", "z", "fx", "fy", "fz"}
            if len(fields) != len(set(fields)) or not required <= set(fields):
                raise ValueError(f"{path}: LAMMPS dump requires unique id,x,y,z,fx,fy,fz columns")
            # shortcut: Cartesian x/y/z only; add box-aware conversion if scaled coordinates are needed.
            for line in atoms:
                row = dict(zip(fields, line.split()))
                force = tuple(float(row[k]) for k in ("fx", "fy", "fz"))
                position = tuple(float(row[k]) for k in ("x", "y", "z"))
                if not all(math.isfinite(v) for v in force + position):
                    raise ValueError(f"{path}: non-finite reference force or position")
                yield step, int(row["id"]) + id_offset, force, position


def analyze(files, reference_files=None, reference_sample=None, id_offset=0,
            position_tolerance=1.e-10):
    reference = {}
    selected = reference_sample
    if reference_files:
        for path in reference_files:
            for sample, atom, force, position in reference_rows(path, id_offset):
                if selected is None:
                    selected = sample
                if sample != selected:
                    continue
                if atom in reference:
                    raise ValueError(f"Duplicate reference atom ID {atom}")
                reference[atom] = (force, position)
        if not reference:
            raise ValueError("Reference sample has no atoms")
    # ID -> [count, mean[3], M2[3], initial position]
    moments = {}
    sample_metrics = []
    expected_ids = None
    total_error2 = 0.0
    merged = heapq.merge(*(rows(path) for path in files), key=lambda row: row[0])
    for sample, group in itertools.groupby(merged, key=lambda row: row[0]):
        seen = set()
        error2 = 0.0
        for _, atom, force, position in group:
            if atom in seen:
                raise ValueError(f"Sample {sample}: duplicate owned ID {atom} (ghost or overlapping shard)")
            seen.add(atom)
            if position is None:
                raise ValueError("Sample CSVs require x,y,z to verify a fixed configuration")
            if atom not in moments:
                moments[atom] = [0, [0.0]*3, [0.0]*3, position]
            state = moments[atom]
            if any(abs(a-b) > position_tolerance for a, b in zip(position, state[3])):
                raise ValueError(f"Sample {sample}: atom {atom} moved; fixed-configuration variance is invalid")
            state[0] += 1
            for axis, value in enumerate(force):
                delta = value - state[1][axis]
                state[1][axis] += delta / state[0]
                state[2][axis] += delta * (value - state[1][axis])
            if reference:
                if atom not in reference:
                    raise ValueError(f"Reference missing atom ID {atom}")
                ref, ref_position = reference[atom]
                if ref_position is not None and any(abs(a-b) > position_tolerance for a, b in zip(position, ref_position)):
                    raise ValueError(f"Reference position mismatch for atom {atom}")
                error2 += sum((a-b)**2 for a, b in zip(force, ref))
        if expected_ids is None:
            expected_ids = seen
            if reference and seen != reference.keys():
                raise ValueError("Reference and sample atom ID sets differ")
        elif seen != expected_ids:
            raise ValueError(f"Sample {sample}: atom ID set differs from the first sample")
        total_error2 += error2
        sample_metrics.append({"sample": sample, "rms_force_error": math.sqrt(error2 / len(seen)) if reference else None})
    if not moments:
        raise ValueError("No force samples")
    n, m = len(moments), len(sample_metrics)
    if any(state[0] != m for state in moments.values()):
        raise ValueError("Unequal number of samples per atom")
    m2_total = sum(sum(state[2]) for state in moments.values())
    population_variance = max(0.0, m2_total / (n*m))
    result = {
        "n_atoms": n, "n_samples": m,
        "rms_definition": "sqrt(sum_samples_atoms |F-Fref|^2 / (M*N)); vector RMS",
        "variance_definition": "mean_atoms trace(force covariance); population divides by M, unbiased by M-1",
        "variance_population": population_variance,
        "variance_unbiased": max(0.0, m2_total / (n*(m-1))) if m > 1 else None,
        "rms_fluctuation": math.sqrt(population_variance),
        "reference_sample": selected if reference else None,
        "reference_positions_checked": bool(reference) and all(p is not None for _, p in reference.values()),
        "samples": sample_metrics,
    }
    if reference:
        ref2 = sum(sum(v*v for v in f) for f, _ in reference.values()) / n
        mse = total_error2 / (n*m)
        bias2 = sum(sum((a-b)**2 for a, b in zip(state[1], reference[atom][0])) for atom, state in moments.items()) / n
        result.update(rms_force_error=math.sqrt(mse), relative_rms_force_error=math.sqrt(mse/ref2) if ref2 else None,
                      rms_bias=math.sqrt(bias2), mse= mse,
                      decomposition_residual=mse-bias2-population_variance)
    if population_variance == 0 and m > 1:
        result["sampling_note"] = "All observed forces identical; verify resampling if a stochastic method was intended."
    return result, moments, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", help="CSV files or quoted glob patterns, all ranks from ONE run")
    parser.add_argument("--reference", nargs="+", help="Reference CSV shards or LAMMPS text custom dump (id x y z fx fy fz); use converged forces in matching units")
    parser.add_argument("--reference-sample", type=int, help="Reference CSV sample or LAMMPS timestep; default: first nonempty sample")
    parser.add_argument("--reference-id-offset", type=int, default=0, help="Explicit offset added to reference IDs; e.g. -1 for 1-based vs 0-based IDs")
    parser.add_argument("--position-tolerance", type=float, default=1.e-10)
    parser.add_argument("--output", type=Path, default=Path("force_statistics.json"))
    parser.add_argument("--per-atom", type=Path, help="Optional per-atom means, unbiased variance trace and bias")
    args = parser.parse_args()
    if not math.isfinite(args.position_tolerance) or args.position_tolerance < 0:
        parser.error("position-tolerance must be finite and nonnegative")
    try:
        result, moments, reference = analyze(expand(args.samples), expand(args.reference) if args.reference else None,
                                             args.reference_sample, args.reference_id_offset, args.position_tolerance)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if args.per_atom:
            with args.per_atom.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "samples", "mean_fx", "mean_fy", "mean_fz", "variance_trace_unbiased", "bias_norm"])
                for atom, (count, mean, m2, _) in sorted(moments.items()):
                    bias = math.sqrt(sum((a-b)**2 for a, b in zip(mean, reference[atom][0]))) if reference else ""
                    writer.writerow([atom, count, *mean, max(0.0, sum(m2)/(count-1)) if count > 1 else "", bias])
        print(args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
