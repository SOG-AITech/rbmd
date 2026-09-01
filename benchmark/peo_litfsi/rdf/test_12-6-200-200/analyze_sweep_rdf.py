#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch RDF analysis for an RBMD parameter sweep.

Place this next to runs/ (i.e. in test_rshell/), then:

    python3 analyze_sweep_rdf.py

For every runs/<tag>/ directory it:
  1. streams the trajectory (default rbmd.trj) and computes one RDF per
     requested atom-type pair (default Li-O_PEO = 10-2, Li-O_TFSI = 10-7),
     normalized exactly the way LAMMPS `compute rdf` normalizes: the density
     of the *second* type in the pair, so the curves are directly comparable
     with the embedded benchmark;
  2. compares each curve against the built-in benchmark and scores it;
  3. writes a point-line figure with every run + the benchmark overlaid and
     the similarity score in the legend;
  4. writes two CSV tables (wide curves + per-run summary with similarity).

The pair definitions, the benchmark, the bins and the trajectory filename are
all command-line options, so the same script works for any later sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

# ---------------------------------------------------------------------------
# Default atom-type pairs. The reference curve is always computed from a
# trajectory through the same code path, so there is no tabulated benchmark.
# ---------------------------------------------------------------------------
DEFAULT_PAIRS = ["Li-O_PEO:10-2", "Li-O_TFSI:10-7"]


# ---------------------------------------------------------------------------
# Pair spec
# ---------------------------------------------------------------------------
@dataclass
class PairSpec:
    name: str
    center_types: frozenset            # "I" types: RDF is measured around these
    neighbor_types: frozenset          # "J" types: their density does the normalizing

    @property
    def all_types(self) -> frozenset:
        return self.center_types | self.neighbor_types


def parse_pair(text: str) -> PairSpec:
    """Parse 'Name:10-2' or 'Name:10-2,7' (comma = several types on one side)."""
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"pair must look like Name:10-2, got {text!r}")
    name, types_part = text.split(":", 1)
    name = name.strip()
    if "-" not in types_part:
        raise argparse.ArgumentTypeError(f"pair must look like Name:10-2, got {text!r}")
    left, right = types_part.split("-", 1)

    def to_set(s: str) -> frozenset:
        out = set()
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                out.add(int(tok))
        if not out:
            raise argparse.ArgumentTypeError(f"empty type list in {text!r}")
        return frozenset(out)

    return PairSpec(name, to_set(left), to_set(right))


# ---------------------------------------------------------------------------
# Trajectory streaming
# ---------------------------------------------------------------------------
def _choose_position_columns(columns, mode):
    wanted = ("x", "y", "z") if mode == "wrapped" else ("xu", "yu", "zu")
    if all(n in columns for n in wanted):
        return columns.index(wanted[0]), columns.index(wanted[1]), columns.index(wanted[2])
    if mode == "unwrapped" and all(n in columns for n in ("x", "y", "z")):
        return columns.index("x"), columns.index("y"), columns.index("z")
    raise ValueError(f"trajectory lacks requested coordinate columns {wanted}; has {columns}")


def _resolve_id_column(columns, wanted_types, element_map):
    """Return (column_index, {token_bytes: type_number}) for identifying atom types.

    Prefers a real 'type' column. Falls back to an 'element' column, which
    LAMMPS writes when the dump uses `dump_modify ... element <labels>`; the
    Nth label there corresponds to atom type N, so the mapping is recoverable.
    """
    if "type" in columns:
        return columns.index("type"), {str(t).encode(): t for t in wanted_types}

    if "element" in columns:
        if not element_map:
            raise ValueError(
                f"trajectory has an 'element' column but no 'type' column, and no element map was "
                f"given. Pass --element-map with the labels from your dump_modify line, in atom-type "
                f"order, e.g. --element-map C2,O,H,C3,N,S,Os,Cf,F,Li . Columns present: {columns}"
            )
        label_of_type = {i + 1: lab for i, lab in enumerate(element_map)}
        lookup = {}
        for t in sorted(wanted_types):
            if t not in label_of_type:
                raise ValueError(
                    f"atom type {t} is outside the element map, which only defines types "
                    f"1..{len(element_map)} ({','.join(element_map)}). Extend --element-map."
                )
            lookup[label_of_type[t].encode()] = t
        return columns.index("element"), lookup

    raise ValueError(f"trajectory has neither a 'type' nor an 'element' column: {columns}")


def scan_frame_offsets(path: Path) -> list:
    """Return the byte offset of every 'ITEM: TIMESTEP' line in the file.

    This is a pure byte scan with no number parsing, so it is far cheaper than a
    real pass over the data. Knowing the offsets lets a caller that only wants
    the last N frames seek straight to them instead of reading through
    everything that comes before.
    """
    offsets = []
    with path.open("rb") as fh:
        pos = 0
        for line in fh:
            if line.startswith(b"ITEM: TIMESTEP"):
                offsets.append(pos)
            pos += len(line)
    return offsets


def iter_frames(path: Path, wanted_types: frozenset, position_mode: str,
                frame_start: int, frame_stop, frame_stride: int, element_map=None,
                seek_to: int = 0, first_frame_index: int = 0):
    """Yield (timestep, box_lengths, {type: Nx3 float32 array}) for selected frames.

    Only atoms whose type is in wanted_types are parsed into arrays; everything
    else is skipped as cheaply as possible. Frames that are not selected have
    their atom block skipped without any float parsing at all.

    Works with both 'type' dumps (RBMD's rbmd.trj) and 'element' dumps
    (LAMMPS `dump custom ... element ...`), so the reference trajectory and the
    sweep trajectories still go through one identical code path.

    seek_to / first_frame_index let a caller jump directly to a known byte
    offset while keeping frame numbering consistent with the whole file.
    """
    import numpy as np

    frame_index = first_frame_index - 1

    with path.open("rb") as fh:
        if seek_to:
            fh.seek(seek_to)
        while True:
            # ---- locate next frame header
            for line in fh:
                if line.startswith(b"ITEM: TIMESTEP"):
                    break
            else:
                return

            frame_index += 1
            timestep = int(fh.readline().strip())

            line = fh.readline()
            if not line.startswith(b"ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"expected NUMBER OF ATOMS, got {line[:80]!r}")
            n_atoms = int(fh.readline().strip())

            line = fh.readline()
            if not line.startswith(b"ITEM: BOX BOUNDS"):
                raise ValueError(f"expected BOX BOUNDS, got {line[:80]!r}")
            bounds = []
            for _ in range(3):
                vals = [float(v) for v in fh.readline().split()]
                if len(vals) > 2:
                    raise ValueError("triclinic boxes are not supported")
                bounds.append((vals[0], vals[1]))
            lengths = tuple(hi - lo for lo, hi in bounds)
            center = tuple(0.5 * (lo + hi) for lo, hi in bounds)

            header = fh.readline()
            if not header.startswith(b"ITEM: ATOMS"):
                raise ValueError(f"expected ITEM: ATOMS, got {header[:80]!r}")
            columns = header.decode(errors="replace").split()[2:]
            type_col, type_bytes = _resolve_id_column(columns, wanted_types, element_map)
            xc, yc, zc = _choose_position_columns(columns, position_mode)
            max_col = max(type_col, xc, yc, zc)

            selected = (
                frame_index >= frame_start
                and (frame_stop is None or frame_index < frame_stop)
                and (frame_index - frame_start) % frame_stride == 0
            )

            if not selected:
                for _ in range(n_atoms):
                    fh.readline()
                continue

            buckets = {t: [] for t in wanted_types}
            for _ in range(n_atoms):
                row = fh.readline()
                f = row.split(None, max_col + 1)
                t = type_bytes.get(f[type_col])
                if t is not None:
                    buckets[t].append((float(f[xc]), float(f[yc]), float(f[zc])))

            arrays = {}
            for t, pts in buckets.items():
                a = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
                if a.size:
                    a -= np.asarray(center, dtype=np.float32)
                arrays[t] = a

            yield timestep, lengths, arrays


# ---------------------------------------------------------------------------
# RDF accumulation (SciPy cKDTree with periodic boxes)
# ---------------------------------------------------------------------------
class RDFAccumulator:
    """g(r) of 'neighbor' atoms around 'center' atoms.

    Normalization matches LAMMPS `compute rdf` for pair (I, J):

        g(r) = <n_J(r)> / (N_I * rho_J * V_shell(r)),   rho_J = N_J / V_box

    Self-pairs are removed when the two type sets are identical, so a
    same-species pair (e.g. O-O) is also handled correctly.
    """

    def __init__(self, edges):
        import numpy as np

        self.edges = np.asarray(edges, dtype=np.float64)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.shell_vol = (4.0 * math.pi / 3.0) * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)
        self.g_sum = np.zeros(len(self.centers), dtype=np.float64)
        self.frames = 0
        self.n_center_last = 0
        self.n_neighbor_last = 0
        self.rho_sum = 0.0          # running sum of the neighbour number density
        self.box_min = None         # shortest box edge seen, for the r_max sanity check

    def add_frame(self, center_pts, neighbor_pts, lengths, same_selection: bool):
        import numpy as np
        from scipy.spatial import cKDTree

        n_c = len(center_pts)
        n_n = len(neighbor_pts)
        if n_c == 0 or n_n == 0:
            return
        L = np.asarray(lengths, dtype=np.float64)
        volume = float(np.prod(L))
        rho = n_n / volume
        if rho <= 0:
            return

        def fold(p):
            q = np.asarray(p, dtype=np.float64) + 0.5 * L
            q = np.mod(q, L)
            # np.mod can return exactly L for tiny negative inputs, and
            # cKDTree(boxsize=L) rejects any coordinate equal to the box length.
            np.clip(q, 0.0, np.nextafter(L, 0.0), out=q)
            return q

        tree_n = cKDTree(fold(neighbor_pts), boxsize=L)
        if same_selection:
            cumulative = tree_n.count_neighbors(tree_n, self.edges, cumulative=True)
        else:
            tree_c = cKDTree(fold(center_pts), boxsize=L)
            cumulative = tree_c.count_neighbors(tree_n, self.edges, cumulative=True)
        # count_neighbors is cumulative, so the self-pairs of a same-species
        # selection all sit at distance 0 in cumulative[0]; np.diff drops them
        # automatically and no extra correction is needed here.
        counts = np.diff(cumulative).astype(np.float64)

        denom = n_c * rho * self.shell_vol
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(denom > 0, counts / denom, 0.0)
        self.g_sum += g
        self.frames += 1
        self.n_center_last = n_c
        self.n_neighbor_last = n_n
        self.rho_sum += rho
        bm = float(np.min(L))
        self.box_min = bm if self.box_min is None else min(self.box_min, bm)

    @property
    def g(self):
        import numpy as np

        if self.frames == 0:
            return np.zeros_like(self.g_sum)
        return self.g_sum / self.frames

    @property
    def rho(self) -> float:
        """Frame-averaged number density of the neighbour species."""
        return self.rho_sum / self.frames if self.frames else 0.0


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def running_coordination(edges, g, rho):
    """Running coordination number n(R) = 4*pi*rho * int_0^R g(r) r^2 dr,
    evaluated at every bin edge.

    Integrating the histogram shell by shell, rather than resampling it, is what
    makes the result insensitive to the bin width: discretisation errors cancel
    in the sum, whereas a peak height exposes them in full.
    """
    import numpy as np

    g = np.asarray(g, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    shell = (e[1:] ** 3 - e[:-1] ** 3) / 3.0
    return e[1:], 4.0 * math.pi * rho * np.cumsum(g * shell)


def coordination_to(edges, g, rho, r_cut):
    """Coordination number integrated to an arbitrary r_cut.

    The bin straddling r_cut is counted pro rata, so the answer does not jump
    when r_cut falls inside a bin instead of on its edge.
    """
    import numpy as np

    g = np.asarray(g, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    total = 0.0
    for i in range(len(g)):
        lo, hi = e[i], e[i + 1]
        if lo >= r_cut:
            break
        hi_eff = min(hi, r_cut)
        total += g[i] * (hi_eff ** 3 - lo ** 3) / 3.0
    return 4.0 * math.pi * rho * total


def first_minimum(centers, g, r_search_max=6.0):
    """Locate the first minimum of g(r) after its first peak.

    A 3-point moving average is used for the search only, never for the reported
    numbers, so that one noisy bin cannot masquerade as the shell boundary.
    Returns None when no clear minimum is found.
    """
    import numpy as np

    centers = np.asarray(centers, dtype=float)
    g = np.asarray(g, dtype=float)
    mask = centers <= r_search_max
    if mask.sum() < 5:
        return None
    c = centers[mask]
    smooth = np.convolve(g[mask], np.ones(3) / 3.0, mode="same")

    peak = int(np.argmax(smooth))
    for i in range(peak + 1, len(smooth) - 1):
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
            return float(c[i])
    return None


def similarity_metrics(g, g_ref) -> dict:
    """Score a computed g(r) against the benchmark.

    Primary score: S = 1 - R, with R the normalized L2 residual

        R = sqrt( sum (g - g_ref)^2 / sum g_ref^2 )

    S = 1 means the curves coincide; S falls off as they deviate and can go
    negative for a badly wrong curve, which is informative rather than a bug.
    R is scale-aware (unlike a correlation coefficient), so a run that gets the
    shape right but the height wrong is correctly penalized -- important here,
    since the whole point is to check that the cheap neighbor/coulomb settings
    reproduce the reference structure quantitatively.

    Also reported, because no single number tells the whole story:
      rmse, mae, max_abs_dev  -- absolute deviations in g units
      pearson_r               -- shape agreement only, ignores scale
      peak_r / peak_g         -- first-peak position and height vs. reference
    """
    import numpy as np

    g = np.asarray(g, dtype=np.float64)
    g_ref = np.asarray(g_ref, dtype=np.float64)

    diff = g - g_ref
    ss_ref = float(np.sum(g_ref ** 2))
    r_factor = math.sqrt(float(np.sum(diff ** 2)) / ss_ref) if ss_ref > 0 else float("nan")
    similarity = 1.0 - r_factor

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))

    if g.std() > 0 and g_ref.std() > 0:
        pearson = float(np.corrcoef(g, g_ref)[0, 1])
    else:
        pearson = float("nan")

    return {
        "similarity": similarity,
        "r_factor": r_factor,
        "rmse": rmse,
        "mae": mae,
        "max_abs_dev": max_abs,
        "pearson_r": pearson,
    }


def peak_info(r, g):
    import numpy as np

    g = np.asarray(g, dtype=np.float64)
    if g.size == 0 or not np.any(np.isfinite(g)):
        return float("nan"), float("nan")
    i = int(np.nanargmax(g))
    return float(r[i]), float(g[i])


# ---------------------------------------------------------------------------
# Core: RDF from one trajectory file
# ---------------------------------------------------------------------------
def rdf_from_trajectory(traj: Path, pairs, edges, position, frame_start, frame_stop, frame_stride,
                        element_map=None, last_frames=None):
    """Accumulate one RDF per pair from a single trajectory file.

    This is the ONLY place RDFs are produced, so the reference trajectory and
    every sweep trajectory are treated bit-for-bit identically: same parser,
    same binning, same normalization, same PBC handling. Any remaining
    difference between the curves is physics, not methodology.

    last_frames=N averages only the final N frames, which is normally what you
    want: the early frames are still relaxing away from the input structure.
    The file is scanned for frame offsets first so the skipped frames are never
    read at all, rather than read and discarded.

    Returns (curves, counts, n_frames_used, n_frames_total).
    """
    import numpy as np

    seek_to = 0
    first_index = 0
    total_frames = None

    if last_frames is not None and last_frames > 0:
        offsets = scan_frame_offsets(traj)
        total_frames = len(offsets)
        if total_frames == 0:
            return {}, {}, 0, 0
        first_index = max(0, total_frames - last_frames)
        seek_to = offsets[first_index]
        frame_start, frame_stop, frame_stride = first_index, None, 1

    wanted = frozenset().union(*[p.all_types for p in pairs])
    accs = {p.name: RDFAccumulator(edges) for p in pairs}

    for _timestep, lengths, arrays in iter_frames(
        traj, wanted, position, frame_start, frame_stop, frame_stride, element_map,
        seek_to, first_index
    ):
        for p in pairs:
            c = np.concatenate([arrays[t] for t in sorted(p.center_types)]) \
                if len(p.center_types) > 1 else arrays[next(iter(p.center_types))]
            n = np.concatenate([arrays[t] for t in sorted(p.neighbor_types)]) \
                if len(p.neighbor_types) > 1 else arrays[next(iter(p.neighbor_types))]
            accs[p.name].add_frame(c, n, lengths, p.center_types == p.neighbor_types)

    curves = {name: acc.g.tolist() for name, acc in accs.items()}
    counts = {name: {"n_center": acc.n_center_last, "n_neighbor": acc.n_neighbor_last,
                     "rho": acc.rho, "box_min": acc.box_min}
              for name, acc in accs.items()}
    frames = max((a.frames for a in accs.values()), default=0)
    if total_frames is None:
        total_frames = frames
    return curves, counts, frames, total_frames


def pairs_from_job(job_pairs):
    return [PairSpec(p["name"], frozenset(p["center"]), frozenset(p["neighbor"]))
            for p in job_pairs]


# ---------------------------------------------------------------------------
# Per-directory worker
# ---------------------------------------------------------------------------
def analyze_one(job: dict) -> dict:
    import numpy as np

    run_dir = Path(job["run_dir"])
    tag = job["tag"]
    pairs = pairs_from_job(job["pairs"])
    edges = np.asarray(job["edges"], dtype=np.float64)

    traj = run_dir / job["traj_name"]
    result = {"tag": tag, "run_dir": str(run_dir), "ok": False, "error": None,
              "curves": {}, "frames": 0, "counts": {}}

    params_file = run_dir / "params.json"
    if params_file.is_file():
        try:
            result["params"] = json.loads(params_file.read_text(encoding="utf-8"))
        except Exception:
            result["params"] = {}
    else:
        result["params"] = {}

    if not traj.is_file():
        result["error"] = f"missing {traj.name}"
        return result

    t0 = time.time()
    try:
        curves, counts, frames, total_frames = rdf_from_trajectory(
            traj, pairs, edges, job["position"],
            job["frame_start"], job["frame_stop"], job["frame_stride"],
            job.get("element_map"), job.get("last_frames"),
        )
    except Exception as exc:  # keep the sweep going if one run is broken
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if frames == 0:
        result["error"] = "no frames selected"
        return result

    result["curves"] = curves
    result["counts"] = counts
    result["frames"] = frames
    result["total_frames"] = total_frames
    result["elapsed_s"] = time.time() - t0
    result["ok"] = True
    return result

def discover_benchmark_traj(explicit, runs_dir_name, sweep_traj_name):
    """Locate the reference trajectory.

    'auto' searches the working directory for the usual LAMMPS dump extensions.
    Auto-discovery only commits when exactly one candidate exists; anything else
    is reported so the choice of reference is never made silently.
    """
    if explicit and explicit != "auto":
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(
                f"reference trajectory not found: {p}\n"
                f"  (looked relative to {Path.cwd()})\n"
                f"  Fix the path with --benchmark-traj, or use --benchmark-traj auto to search "
                f"the current directory, or --benchmark-mode table to score against the embedded "
                f"LAMMPS rdf.dat instead."
            )
        return p

    patterns = ("*.lammpstrj", "dump*", "*.dump", "*.trj")
    seen = {}
    for pat in patterns:
        for c in Path(".").glob(pat):
            if not c.is_file():
                continue
            if c.name == sweep_traj_name:      # that is a sweep output, not the reference
                continue
            seen[c.resolve()] = c
    candidates = sorted(seen.values(), key=lambda p: p.name)

    if len(candidates) == 1:
        print(f"auto-discovered reference trajectory: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"no reference trajectory found in {Path.cwd()}\n"
            f"  (searched for: {', '.join(patterns)}, excluding '{sweep_traj_name}' and {runs_dir_name}/)\n"
            f"  Point at it explicitly with --benchmark-traj <file>, or score against the embedded "
            f"LAMMPS rdf.dat with --benchmark-mode table."
        )
    listing = "\n".join(f"    {c.name}  ({c.stat().st_size / 1e6:.1f} MB)" for c in candidates)
    raise SystemExit(
        f"several possible reference trajectories found in {Path.cwd()}:\n{listing}\n"
        f"  Pick one with --benchmark-traj <file>."
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(r, bench, results, pairs, out_png, ylabel_fmt="g(r)   [{name}]",
              hline=None, bench_label="benchmark", title_prefix="", vlines=None,
              xlim=None, annotate=None):
    """Draw one panel per pair: the reference plus every sweep run.

    Everything here is computed from a trajectory through the same code path, so
    every curve is drawn the same way -- a bare line, no markers, which keeps
    overlapping curves readable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    good = [res for res in results if res["ok"]]
    n_runs = max(len(good), 1)
    n_pairs = len(pairs)

    legend_outside = n_runs > 8
    fig_w = 13.5 if legend_outside else 11.0
    fig, axes = plt.subplots(n_pairs, 1, figsize=(fig_w, 5.0 * n_pairs), squeeze=False)

    if n_runs <= 10:
        colors = [plt.get_cmap("tab10")(i) for i in range(n_runs)]
    elif n_runs <= 20:
        colors = [plt.get_cmap("tab20")(i) for i in range(n_runs)]
    else:
        colors = [plt.get_cmap("viridis")(i / max(n_runs - 1, 1)) for i in range(n_runs)]

    # With markers gone, many overlapping solid lines get hard to tell apart,
    # so dash patterns take over that job once the sweep is large.
    linestyles = ["-", "--", "-.", ":"] if n_runs > 10 else ["-"]

    for ax, p in zip(axes[:, 0], pairs):
        entry = bench.get(p.name)
        if entry is not None:
            bx, by = entry
            ax.plot(bx, by, ls="-", color="black", lw=2.6, zorder=10, label=bench_label)

        for i, res in enumerate(good):
            curve = res["plot"].get(p.name)
            if curve is None:
                continue
            cx, cy = curve
            score = res["scores"].get(p.name, {}).get("similarity")
            label = res["tag"] if score is None else f"{res['tag']}  (S={score:.3f})"
            ax.plot(cx, cy, ls=linestyles[i % len(linestyles)], color=colors[i],
                    lw=1.5, alpha=0.9, label=label)

        if hline is not None:
            ax.axhline(hline, color="grey", lw=0.8, ls=":", zorder=0)

        vx = (vlines or {}).get(p.name)
        if vx is not None:
            ax.axvline(vx, color="grey", lw=1.2, ls="--", zorder=1)
            ax.annotate(f"first-shell cut\n r = {vx:.2f} Å", xy=(vx, 0.97),
                        xycoords=("data", "axes fraction"), fontsize=8, color="dimgrey",
                        ha="left" if vx < 0.6 * float(r[-1]) else "right",
                        va="top", xytext=(4 if vx < 0.6 * float(r[-1]) else -4, 0),
                        textcoords="offset points")

        note = (annotate or {}).get(p.name)
        if note:
            ax.annotate(note, xy=(0.015, 0.03), xycoords="axes fraction", fontsize=8,
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
        ax.set_xlabel("r (Å)")
        ax.set_ylabel(ylabel_fmt.format(name=p.name))
        types_txt = (f"{sorted(p.center_types)}-{sorted(p.neighbor_types)}"
                     .replace("[", "").replace("]", "").replace(" ", ""))
        ax.set_title(f"{title_prefix}{p.name}   (types {types_txt})")
        if legend_outside:
            ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
        else:
            ax.legend(fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.6)
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(float(r[0]) - 0.3, float(r[-1]) + 0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import numpy as np

    ap = argparse.ArgumentParser(
        description="Compute RDFs for every run in a sweep, compare against a benchmark, plot and tabulate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--runs-dir", default="runs", help="directory holding one subdirectory per run")
    ap.add_argument("--outdir", default="rdf_analysis", help="where figures and tables are written")
    ap.add_argument("--traj-name", default="rbmd.trj", help="trajectory filename inside each run directory")
    ap.add_argument("--pair", action="append", default=None, metavar="NAME:I-J",
                    help=("atom-type pair to analyze, repeatable. 'I' are the central atoms, "
                          "'J' the neighbors whose density normalizes g(r). Several types per "
                          "side allowed with commas, e.g. Li-Oall:10-2,7. "
                          f"default: {' and '.join(DEFAULT_PAIRS)}"))
    ap.add_argument("--last-frames", type=int, default=5,
                    help=("average only the final N frames of each sweep trajectory, since the early "
                          "frames are still relaxing away from the input structure. Set to 0 to use "
                          "the explicit --frame-start/--frame-stop/--frame-stride range instead"))
    ap.add_argument("--benchmark-last-frames", type=int, default=None,
                    help="same, for the reference trajectory (default: follow --last-frames)")
    ap.add_argument("--element-map", default="C2,O,H,C3,N,S,Os,Cf,F,Li",
                    help=("element labels in atom-type order, used only for trajectories that carry an "
                          "'element' column instead of 'type'. Must match the dump_modify element line: "
                          "the Nth label is atom type N. Set to an empty string to disable"))
    ap.add_argument("--benchmark-traj", default="auto",
                    help=("reference trajectory, or 'auto' to find it in the current directory. "
                          "Its RDF is computed with the SAME code path as the sweep runs, so method "
                          "differences cancel out and only physics remains. Multi-frame files are "
                          "averaged over every frame"))
    ap.add_argument("--benchmark-frame-start", type=int, default=0,
                    help=("first frame of the reference trajectory to use. Deliberately separate from "
                          "--frame-start so that skipping equilibration in the sweep runs does not "
                          "silently discard reference frames; by default every reference frame is averaged"))
    ap.add_argument("--benchmark-frame-stop", type=int, default=None,
                    help="stop before this frame of the reference trajectory (default: use all of them)")
    ap.add_argument("--cn-rcut", action="append", default=None, metavar="[NAME=]VALUE",
                    help=("first-shell integration limit for the coordination number, in Angstrom. "
                          "Repeatable; a bare number applies to every pair, NAME=VALUE targets one. "
                          "Default: the first minimum of the reference g(r), which guarantees that "
                          "all runs are integrated to the same limit"))
    ap.add_argument("--cn-plot-rmax", type=float, default=None,
                    help=("x-axis limit of the coordination-number figures. n(r) grows as r^3 once "
                          "g(r) reaches 1, so the default zooms to 2.5x the first-shell cut instead "
                          "of showing bulk counting all the way to --r-max"))
    ap.add_argument("--r-max", type=float, default=18.0)
    ap.add_argument("--r-min", type=float, default=0.0)
    ap.add_argument("--dr", "--bin-width", dest="dr", type=float, default=0.25,
                    help=("RDF bin width in Angstrom. The Li-O first shell is only ~0.3 A wide, so "
                          "0.5 barely resolves it with two points; 0.25 or finer is usually better"))
    ap.add_argument("--bins", type=int, default=None, help="explicit bin count (overrides --dr)")
    ap.add_argument("--position", choices=("wrapped", "unwrapped"), default="wrapped")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-stop", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="use every Nth frame; with only ~10 frames per trajectory, keep this at 1")
    ap.add_argument("--jobs", type=int, default=4, help="run directories analyzed in parallel")
    ap.add_argument("--only", default=None, help="analyze only run tags containing this substring")
    args = ap.parse_args(argv)

    # ---- bins
    if args.bins is None:
        span = args.r_max - args.r_min
        nbins = int(round(span / args.dr))
        if nbins < 1:
            raise SystemExit("bad --r-max/--r-min/--dr combination")
        if abs(nbins * args.dr - span) > 1e-9 * max(1.0, span):
            print(f"warning: span {span} is not a multiple of dr {args.dr}; using {nbins} bins",
                  file=sys.stderr)
        args.bins = nbins
    edges = np.linspace(args.r_min, args.r_max, args.bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    # ---- pairs
    pair_texts = args.pair if args.pair else DEFAULT_PAIRS
    pairs = [parse_pair(t) for t in pair_texts]

    element_map = [s.strip() for s in args.element_map.split(",") if s.strip()] if args.element_map else []

    # ---- reference computed from a trajectory through the identical code path
    bench_traj = discover_benchmark_traj(args.benchmark_traj, args.runs_dir, args.traj_name)
    print(f"computing reference RDF from {bench_traj} (same parser, bins and normalization "
          f"as the sweep runs)...", flush=True)
    t_ref = time.time()
    bench_last = args.benchmark_last_frames if args.benchmark_last_frames is not None else args.last_frames
    try:
        traj_curves, ref_counts, ref_frames, ref_total = rdf_from_trajectory(
            bench_traj, pairs, edges, args.position,
            args.benchmark_frame_start, args.benchmark_frame_stop, 1,
            element_map, bench_last,
        )
    except Exception as exc:
        raise SystemExit(f"could not read the reference trajectory {bench_traj}:\n  "
                         f"{type(exc).__name__}: {exc}")
    if ref_frames == 0:
        raise SystemExit(f"no frames could be read from {bench_traj}")
    bench_curves = {k: np.asarray(v, dtype=float) for k, v in traj_curves.items()}
    for name, c in ref_counts.items():
        print(f"  reference {name}: n_center={c['n_center']} n_neighbor={c['n_neighbor']}")
    el = time.time() - t_ref
    if bench_last and ref_total > ref_frames:
        print(f"  last {ref_frames} of {ref_total} frames averaged in {el:.1f}s", flush=True)
    elif bench_last and ref_total <= bench_last:
        print(f"  {ref_frames} frame(s) averaged in {el:.1f}s -- the file only has {ref_total}, "
              f"fewer than the {bench_last} requested", flush=True)
    else:
        print(f"  {ref_frames} frame(s) averaged in {el:.1f}s", flush=True)
    bench_label = f"benchmark ({bench_traj.name})"
    _boxes = [c.get("box_min") for c in ref_counts.values() if c.get("box_min")]
    ref_box_half = 0.5 * min(_boxes) if _boxes else None

    # ---- r_max sanity: g(r) past half the box length counts periodic images of
    # the same atom and the 4*pi*r^2 shell normalisation no longer applies.
    if ref_box_half is not None and args.r_max > ref_box_half + 1e-9:
        print(f"warning: r_max={args.r_max} exceeds half the reference box length "
              f"({ref_box_half:.2f} A). Beyond that radius g(r) is not meaningful: the shell "
              f"is no longer fully inside one periodic image. Lower --r-max to <= {ref_box_half:.2f}, "
              f"or use a larger box.", file=sys.stderr)

    # ---- first-shell integration limits, taken from the reference so that every
    # run is integrated to exactly the same radius
    cn_rcut = {}
    overrides_all = None
    overrides_named = {}
    for item in (args.cn_rcut or []):
        if "=" in item:
            k, v = item.split("=", 1)
            overrides_named[k.strip()] = float(v)
        else:
            overrides_all = float(item)
    for p_ in pairs:
        if p_.name in overrides_named:
            cn_rcut[p_.name] = overrides_named[p_.name]
        elif overrides_all is not None:
            cn_rcut[p_.name] = overrides_all
        else:
            rmin = first_minimum(r_centers, bench_curves.get(p_.name, []))
            if rmin is None:
                print(f"warning: could not locate a first minimum for '{p_.name}'; "
                      f"set it manually with --cn-rcut {p_.name}=<r>. Skipping its coordination number.",
                      file=sys.stderr)
            cn_rcut[p_.name] = rmin
    print("\nfirst-shell integration limits (from the reference g(r), shared by all runs):")
    for p_ in pairs:
        rc = cn_rcut.get(p_.name)
        print(f"  {p_.name}: r_cut = {rc:.3f} A" if rc else f"  {p_.name}: unavailable")
    print()

    # ---- discover runs
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"runs directory not found: {runs_dir}")
    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if args.only:
        run_dirs = [d for d in run_dirs if args.only in d.name]
    if not run_dirs:
        raise SystemExit(f"no run subdirectories found under {runs_dir}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = [{
        "run_dir": str(d),
        "tag": d.name,
        "traj_name": args.traj_name,
        "pairs": [{"name": p.name, "center": sorted(p.center_types),
                   "neighbor": sorted(p.neighbor_types)} for p in pairs],
        "edges": edges.tolist(),
        "position": args.position,
        "frame_start": args.frame_start,
        "frame_stop": args.frame_stop,
        "frame_stride": args.frame_stride,
        "element_map": element_map,
        "last_frames": args.last_frames,
    } for d in run_dirs]

    print(f"analyzing {len(jobs)} run(s) from {runs_dir}/ with {args.jobs} parallel worker(s)")
    print(f"pairs: " + ", ".join(f"{p.name}={sorted(p.center_types)}-{sorted(p.neighbor_types)}" for p in pairs))
    print(f"bins: {args.bins} over [{args.r_min}, {args.r_max}] (width {(args.r_max-args.r_min)/args.bins:g})")
    print()

    t0 = time.time()
    if args.jobs > 1 and len(jobs) > 1:
        with Pool(processes=min(args.jobs, len(jobs))) as pool:
            results = pool.map(analyze_one, jobs)
    else:
        results = [analyze_one(j) for j in jobs]

    # ---- score
    for res in results:
        res["scores"] = {}
        if not res["ok"]:
            print(f"  [FAIL] {res['tag']}: {res['error']}")
            continue
        tot = res.get("total_frames", res["frames"])
        fr_txt = f"{res['frames']}/{tot} frames" if tot != res["frames"] else f"{res['frames']} frame(s)"
        msg = [f"  [ok]   {res['tag']}: {fr_txt}, {res.get('elapsed_s', 0):.1f}s"]
        for p in pairs:
            g = res["curves"].get(p.name)
            ref = bench_curves.get(p.name)
            if g is None or ref is None:
                continue
            m = similarity_metrics(g, ref)
            pr, pg = peak_info(r_centers, g)
            rr, rg = peak_info(r_centers, ref)
            m.update({"peak_r": pr, "peak_g": pg, "ref_peak_r": rr, "ref_peak_g": rg,
                      "peak_r_dev": pr - rr, "peak_g_dev": pg - rg})
            res["scores"][p.name] = m
            msg.append(f"{p.name} S={m['similarity']:.3f}")
        print("  ".join(msg))

    print(f"\ntotal wall time: {time.time() - t0:.1f}s")

    good = [r for r in results if r["ok"]]
    if not good:
        raise SystemExit("no run produced a usable RDF; nothing to plot")

    # ---- coordination numbers (dr-insensitive: an integral, not a peak value)
    cn_edges, _ = running_coordination(edges, bench_curves[pairs[0].name],
                                       ref_counts[pairs[0].name]["rho"])
    bench_cn_curve = {}
    bench_cn_shell = {}
    for p in pairs:
        rho = ref_counts[p.name]["rho"]
        _, n_of_r = running_coordination(edges, bench_curves[p.name], rho)
        bench_cn_curve[p.name] = n_of_r
        rc = cn_rcut.get(p.name)
        bench_cn_shell[p.name] = (coordination_to(edges, bench_curves[p.name], rho, rc)
                                  if rc else None)

    for res in good:
        res["cn_curve"] = {}
        res["cn_shell"] = {}
        for p in pairs:
            g = res["curves"].get(p.name)
            if g is None:
                continue
            rho = res["counts"][p.name]["rho"]
            _, n_of_r = running_coordination(edges, g, rho)
            res["cn_curve"][p.name] = n_of_r
            rc = cn_rcut.get(p.name)
            res["cn_shell"][p.name] = coordination_to(edges, g, rho, rc) if rc else None

    # ---- output directories: RDF and coordination number kept apart
    rdf_dir = outdir / "rdf"
    cn_dir = outdir / "cn"
    rdf_dir.mkdir(parents=True, exist_ok=True)
    cn_dir.mkdir(parents=True, exist_ok=True)

    def write_wide(path, x, xlabel, bench_map, run_key):
        header = [xlabel]
        columns = [np.asarray(x, dtype=float)]
        for p in pairs:
            if p.name in bench_map:
                header.append(f"benchmark__{p.name}")
                columns.append(np.asarray(bench_map[p.name], dtype=float))
        for res in good:
            for p in pairs:
                v = res.get(run_key, {}).get(p.name)
                if v is not None:
                    header.append(f"{res['tag']}__{p.name}")
                    columns.append(np.asarray(v, dtype=float))
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for i in range(len(columns[0])):
                w.writerow([f"{col[i]:.6g}" for col in columns])

    write_wide(rdf_dir / "rdf_curves_wide.csv", r_centers, "r", bench_curves, "curves")
    write_wide(cn_dir / "cn_curves_wide.csv", cn_edges, "r", bench_cn_curve, "cn_curve")

    # ---- per-run summary tables
    param_keys = sorted({k for res in results for k in res.get("params", {})})
    metric_keys = ["similarity", "r_factor", "rmse", "mae", "max_abs_dev", "pearson_r",
                   "peak_r", "peak_g", "ref_peak_r", "ref_peak_g", "peak_r_dev", "peak_g_dev"]
    with (rdf_dir / "rdf_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "pair", "status", "frames_used", "frames_total", "n_center", "n_neighbor"]
                   + param_keys + metric_keys)
        for res in results:
            for p in pairs:
                m = res.get("scores", {}).get(p.name, {})
                counts = res.get("counts", {}).get(p.name, {})
                row = [res["tag"], p.name,
                       "ok" if res["ok"] else f"FAIL: {res['error']}",
                       res.get("frames", 0), res.get("total_frames", ""),
                       counts.get("n_center", ""), counts.get("n_neighbor", "")]
                row += [res.get("params", {}).get(k, "") for k in param_keys]
                row += [f"{m[k]:.6g}" if k in m else "" for k in metric_keys]
                w.writerow(row)

    # Fractions only make sense between pairs that share the same central atoms,
    # e.g. how the Li first shell is split between PEO and TFSI oxygens.
    center_groups = {}
    for p in pairs:
        center_groups.setdefault(tuple(sorted(p.center_types)), []).append(p.name)

    def shell_total(shells, group):
        vals = [shells.get(n) for n in group]
        return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None

    with (cn_dir / "cn_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "pair", "status", "frames_used", "frames_total", "r_cut",
                    "n_first_shell", "benchmark_n", "deviation", "rel_deviation_percent",
                    "group_total_n", "fraction_of_group"] + param_keys)
        for res in results:
            for p in pairs:
                grp = center_groups[tuple(sorted(p.center_types))]
                n = res.get("cn_shell", {}).get(p.name)
                ref_n = bench_cn_shell.get(p.name)
                tot = shell_total(res.get("cn_shell", {}), grp) if res["ok"] else None
                dev = (n - ref_n) if (n is not None and ref_n is not None) else None
                rel = (100.0 * dev / ref_n) if (dev is not None and ref_n) else None
                frac = (n / tot) if (n is not None and tot) else None
                rc = cn_rcut.get(p.name)
                row = [res["tag"], p.name,
                       "ok" if res["ok"] else f"FAIL: {res['error']}",
                       res.get("frames", 0), res.get("total_frames", ""),
                       f"{rc:.4g}" if rc else "",
                       f"{n:.4f}" if n is not None else "",
                       f"{ref_n:.4f}" if ref_n is not None else "",
                       f"{dev:+.4f}" if dev is not None else "",
                       f"{rel:+.2f}" if rel is not None else "",
                       f"{tot:.4f}" if tot is not None else "",
                       f"{frac:.4f}" if frac is not None else ""]
                row += [res.get("params", {}).get(k, "") for k in param_keys]
                w.writerow(row)

    # ---- console: RDF ranking
    print("\nRDF similarity ranking (higher is better, 1.0 = identical to the reference):")
    for p in pairs:
        if p.name not in bench_curves:
            continue
        ranked = sorted((res for res in good if p.name in res["scores"]),
                        key=lambda res: res["scores"][p.name]["similarity"], reverse=True)
        print(f"  {p.name}:")
        for res in ranked:
            m = res["scores"][p.name]
            print(f"    S={m['similarity']:+.4f}  rmse={m['rmse']:.4f}  "
                  f"max|dg|={m['max_abs_dev']:.4f}  peak dr={m['peak_r_dev']:+.2f}Å  {res['tag']}")

    # ---- console: coordination numbers, ranked by how far they sit from the reference
    print("\nFirst-shell coordination number (an integral, so it barely moves with dr):")
    for p in pairs:
        rc = cn_rcut.get(p.name)
        ref_n = bench_cn_shell.get(p.name)
        if rc is None or ref_n is None:
            continue
        print(f"  {p.name}  (integrated to {rc:.2f} Å, reference n = {ref_n:.3f}):")
        ranked = sorted((res for res in good if res.get("cn_shell", {}).get(p.name) is not None),
                        key=lambda res: abs(res["cn_shell"][p.name] - ref_n))
        for res in ranked:
            n = res["cn_shell"][p.name]
            print(f"    n={n:6.3f}   dev={n - ref_n:+6.3f} ({100 * (n - ref_n) / ref_n:+6.2f}%)   {res['tag']}")

    for group_types, group in center_groups.items():
        if len(group) < 2:
            continue
        ref_tot = shell_total(bench_cn_shell, group)
        if not ref_tot:
            continue
        types_txt = ",".join(str(x) for x in group_types)
        print(f"\n  Speciation of the type-{types_txt} first shell "
              f"(reference total n = {ref_tot:.3f}):")
        head = "    %-22s %8s" % ("run", "total")
        for nm in group:
            head += " %14s" % nm
        print(head)
        ref_line = "    %-22s %8.3f" % ("benchmark", ref_tot)
        for nm in group:
            v = bench_cn_shell.get(nm)
            ref_line += " %13.1f%%" % (100 * v / ref_tot) if v is not None else " %14s" % "-"
        print(ref_line)
        for res in good:
            tot = shell_total(res.get("cn_shell", {}), group)
            if not tot:
                continue
            line = "    %-22s %8.3f" % (res["tag"], tot)
            for nm in group:
                v = res.get("cn_shell", {}).get(nm)
                line += " %13.1f%%" % (100 * v / tot) if v is not None else " %14s" % "-"
            print(line)

    # ---- figures
    for res in good:
        res["plot"] = {nm: (r_centers, np.asarray(v, dtype=float))
                       for nm, v in res["curves"].items()}
    make_plot(r_centers, {k: (r_centers, np.asarray(v, dtype=float)) for k, v in bench_curves.items()},
              results, pairs, rdf_dir / "rdf_comparison.png",
              ylabel_fmt="g(r)   [{name}]", hline=1.0, bench_label=bench_label)

    # n(r) grows as (4/3)*pi*rho*r^3 once g(r) settles at 1, so plotting it all the
    # way to r_max buries the first shell under bulk counting. Show only the region
    # where the coordination number carries information.
    cuts = [v for v in cn_rcut.values() if v]
    auto_rmax = min(float(args.r_max), 2.5 * max(cuts)) if cuts else float(args.r_max)
    cn_xmax = args.cn_plot_rmax if args.cn_plot_rmax is not None else auto_rmax
    cn_xlim = (0.0, cn_xmax)

    cn_notes = {}
    for p_ in pairs:
        ref_n = bench_cn_shell.get(p_.name)
        if ref_n is not None:
            cn_notes[p_.name] = f"reference first-shell n = {ref_n:.3f}"

    for res in good:
        res["plot"] = {nm: (cn_edges, np.asarray(v, dtype=float))
                       for nm, v in res.get("cn_curve", {}).items()}
    make_plot(cn_edges, {k: (cn_edges, np.asarray(v, dtype=float)) for k, v in bench_cn_curve.items()},
              results, pairs, cn_dir / "cn_comparison.png",
              ylabel_fmt="n(r)   [{name}]", hline=None, bench_label=bench_label,
              title_prefix="running coordination number: ",
              vlines=cn_rcut, xlim=cn_xlim, annotate=cn_notes)

    # Difference against the reference: the curves above sit almost on top of each
    # other, and the gap between them is the quantity actually being judged.
    for res in good:
        res["plot"] = {}
        for nm, v in res.get("cn_curve", {}).items():
            if nm in bench_cn_curve:
                res["plot"][nm] = (cn_edges, np.asarray(v, dtype=float) - bench_cn_curve[nm])
    make_plot(cn_edges, {k: (cn_edges, np.zeros_like(cn_edges)) for k in bench_cn_curve},
              results, pairs, cn_dir / "cn_difference.png",
              ylabel_fmt="n(r) - n_ref(r)   [{name}]", hline=0.0, bench_label="reference (zero)",
              title_prefix="coordination number difference: ",
              vlines=cn_rcut, xlim=cn_xlim)

    print(f"\nwrote:")
    for f in (rdf_dir / "rdf_comparison.png", rdf_dir / "rdf_curves_wide.csv",
              rdf_dir / "rdf_summary.csv", cn_dir / "cn_comparison.png",
              cn_dir / "cn_difference.png", cn_dir / "cn_curves_wide.csv",
              cn_dir / "cn_summary.csv"):
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
