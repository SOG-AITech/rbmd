#!/usr/bin/env python3
import csv
import pathlib
import re
import sys


ROOT = pathlib.Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/public/home/acymkxdx47/rbmd/build/lammps_kokkos_pppm_benchmark_large_lj8_water9_sync_20260812"
)


def parse_metadata(path):
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle, delimiter="\t"))


def section_time(text, name):
    match = re.search(
        rf"(?m)^{re.escape(name)}\s*\|\s*([0-9.eE+-]+)\s*\|", text
    )
    return float(match.group(1)) if match else None


def find_job_id(case_dir, submitted):
    key = str(case_dir)
    return submitted.get(key, "")


submitted = {}
manifest = ROOT / "submitted_jobs.tsv"
if manifest.exists():
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            submitted[row["case_dir"]] = row["job_id"]

rows = []
for meta_path in sorted(ROOT.glob("*/*/nodes_*/case_metadata.tsv")):
    meta = parse_metadata(meta_path)
    case_dir = meta_path.parent
    log_path = case_dir / "log.lammps"
    row = dict(meta)
    row["job_id"] = find_job_id(case_dir, submitted)
    runtime_path = case_dir / "runtime_metadata.txt"
    if runtime_path.exists():
        runtime_text = runtime_path.read_text(errors="replace")
        runtime_job = re.search(r"(?m)^job_id=(\d+)\s*$", runtime_text)
        if runtime_job:
            row["job_id"] = runtime_job.group(1)
    row["status"] = "missing"
    row["pair_total_s"] = ""
    row["kspace_total_s"] = ""
    row["pair_s_per_step"] = ""
    row["kspace_s_per_step"] = ""
    row["force_kernel_s_per_step"] = ""
    row["loop_total_s"] = ""
    row["loop_s_per_step"] = ""
    row["neighbor_builds"] = ""
    row["case_dir"] = str(case_dir)

    if log_path.exists():
        text = log_path.read_text(errors="replace")
        pair = section_time(text, "Pair")
        kspace = section_time(text, "Kspace")
        loop = re.search(
            r"Loop time of\s+([0-9.eE+-]+)\s+on\s+\d+\s+procs\s+"
            r"for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms",
            text,
        )
        builds = re.search(r"Neighbor list builds\s*=\s*(\d+)", text)
        steps = int(meta["steps"])
        if pair is not None and kspace is not None and loop:
            row["status"] = "complete"
            row["pair_total_s"] = f"{pair:.9f}"
            row["kspace_total_s"] = f"{kspace:.9f}"
            row["pair_s_per_step"] = f"{pair / steps:.12g}"
            row["kspace_s_per_step"] = f"{kspace / steps:.12g}"
            row["force_kernel_s_per_step"] = f"{(pair + kspace) / steps:.12g}"
            row["loop_total_s"] = f"{float(loop.group(1)):.9f}"
            row["loop_s_per_step"] = f"{float(loop.group(1)) / steps:.12g}"
            row["neighbor_builds"] = builds.group(1) if builds else ""
            row["atoms"] = loop.group(3)
        elif "ERROR:" in text:
            row["status"] = "failed"
        else:
            row["status"] = "running_or_incomplete"
    rows.append(row)

fieldnames = [
    "system",
    "mode",
    "gpus",
    "nodes",
    "gpus_per_node",
    "atoms",
    "steps",
    "pair_cutoff",
    "kspace_accuracy",
    "newton",
    "replicate_x",
    "replicate_y",
    "replicate_z",
    "job_id",
    "status",
    "pair_total_s",
    "kspace_total_s",
    "pair_s_per_step",
    "kspace_s_per_step",
    "force_kernel_s_per_step",
    "loop_total_s",
    "loop_s_per_step",
    "neighbor_builds",
    "case_dir",
]

output = ROOT / "results.tsv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(output)
print(f"complete={sum(row['status'] == 'complete' for row in rows)}/{len(rows)}")
