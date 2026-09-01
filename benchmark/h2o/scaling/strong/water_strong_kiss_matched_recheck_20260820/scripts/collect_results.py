#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path

ROOT = Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_kiss_matched_recheck_20260820")
LOOP = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms")
SECTION = lambda name: re.compile(rf"^{name}\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)", re.M)
TIMING = re.compile(r"│\s*(Neighbor-List|Short-Range|Long-Range|Bond|Angle)\s*│\s*([0-9.eE+-]+)\s*│")
GRID = re.compile(r"MPI grid dimensions:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

manifest = list(csv.DictReader((ROOT / "manifest.tsv").open(), delimiter="\t"))
submitted = {row["case_dir"]: row["job_id"] for row in csv.DictReader((ROOT / "submitted_jobs.tsv").open(), delimiter="\t")}
out = []
for row in manifest:
    case_dir = Path(row["case_dir"])
    result = {**row, "job_id": submitted.get(row["case_dir"], ""), "status": "not_submitted"}
    if result["job_id"]:
        result["status"] = "queued_or_running"
    meta = case_dir / "runtime_metadata.txt"
    if meta.exists():
        md = dict(line.split("=", 1) for line in meta.read_text().splitlines() if "=" in line)
        result["exit_code"] = md.get("exit_code", "")
        result["elapsed_s"] = md.get("elapsed_s", "")
        result["hostlist"] = md.get("hostlist", "")
        result["metadata_ok"] = (
            md.get("expected_atoms") == row["atoms"]
            and md.get("steps") == row["steps"]
            and md.get("processor_grid") == row["processor_grid"]
            and md.get("replicate") == "6x6x6"
        )
    record = case_dir / "slurm_job_record.txt"
    result["workdir_ok"] = bool(record.exists() and f"WorkDir={case_dir}" in record.read_text(errors="replace"))
    rank_map = case_dir / "rank_map.tsv"
    if rank_map.exists():
        rank_lines = [line for line in rank_map.read_text().splitlines() if line.strip()]
        result["rank_map_ok"] = len(rank_lines) == int(row["dcus"])

    if row["method"].startswith("LAMMPS"):
        log = case_dir / "log.lammps"
        if log.exists():
            text = log.read_text(errors="replace")
            loop = list(LOOP.finditer(text))
            if loop:
                result["loop_time"], result["loop_procs"], result["loop_steps"], result["loop_atoms"] = loop[-1].groups()
            for section in ("Pair", "Kspace", "Comm", "Neigh", "Modify"):
                matches = list(SECTION(section).finditer(text))
                if matches:
                    minimum, average, maximum = matches[-1].groups()
                    key = section.lower()
                    result[f"{key}_avg"] = average
                    result[f"{key}_max"] = maximum
            mesh = re.findall(r"^\s*grid =\s*(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.M)
            result["pppm_mesh"] = "x".join(mesh[-1]) if mesh else ""
            px, py, pz = row["processor_grid"].split("x")
            result["processor_ok"] = bool(re.search(rf"^processors\s+{px}\s+{py}\s+{pz}\s*$", text, re.M))
            result["kiss_fft"] = "using double precision KISS FFT" in text
            result["validated_content"] = (
                result.get("loop_procs") == row["dcus"]
                and result.get("loop_steps") == row["steps"]
                and result.get("loop_atoms") == row["atoms"]
                and result.get("processor_ok")
                and result.get("kiss_fft")
                and all(result.get(key) for key in ("pair_avg", "pair_max", "kspace_avg", "kspace_max"))
            )
    else:
        log = case_dir / "rbmd.log"
        slurm_out = next(case_dir.glob("slurm-*.out"), None)
        if log.exists():
            text = log.read_text(errors="replace")
            rep = re.search(r"replicate_sampling\s*:\s*\(([^)]+)\)", text)
            result["finish"] = "Finish" in text
            result["step1000_seen"] = bool(re.search(r"\|\s*1000\s*\|", text))
            result["observed_replicate"] = rep.group(1).replace(" ", "") if rep else ""
            for key, value in TIMING.findall(text):
                result[key.lower().replace("-", "_")] = value
        if slurm_out:
            match = GRID.search(slurm_out.read_text(errors="replace"))
            result["observed_grid"] = "x".join(match.groups()) if match else ""
        config = json.loads((case_dir / "run.json").read_text())
        source = json.loads((case_dir / "source_run.json").read_text())
        neighbor = config["hyper_parameters"]["neighbor"]
        coulomb = config["hyper_parameters"]["coulomb"]
        result["parameter_ok"] = (
            float(neighbor.get("cut_off")) == 9
            and float(neighbor.get("r_core")) == 5
            and neighbor.get("neighbor_sample_num") == 30
            and coulomb.get("coulomb_sample_num") == 200
            and config["execution"].get("num_steps") == 1000
        )
        result["validated_content"] = (
            result.get("finish")
            and result.get("step1000_seen")
            and result.get("observed_replicate") == "6,6,6"
            and result.get("observed_grid") == row["processor_grid"]
            and result.get("parameter_ok")
            and config == source
            and all(result.get(key) for key in ("neighbor_list", "short_range", "long_range", "bond", "angle"))
        )

    if (
        result.get("exit_code") == "0"
        and result.get("metadata_ok")
        and result.get("workdir_ok")
        and result.get("rank_map_ok")
        and result.get("validated_content")
    ):
        result["status"] = "validated_complete"
        (case_dir / "VALIDATED_COMPLETE").touch()
    elif result.get("exit_code"):
        result["status"] = "failed_or_invalid"
    out.append(result)

fields = [
    "job_id", "order", "method", "name", "nodes", "dcus", "atoms", "replicate", "steps",
    "processor_grid", "local_replica_ratio", "status", "exit_code", "elapsed_s", "hostlist",
    "metadata_ok", "workdir_ok", "rank_map_ok", "loop_time", "loop_procs", "loop_steps", "loop_atoms",
    "pppm_mesh", "processor_ok", "kiss_fft", "pair_avg", "pair_max", "kspace_avg", "kspace_max",
    "comm_avg", "comm_max", "neigh_avg", "neigh_max", "modify_avg", "modify_max",
    "finish", "step1000_seen", "observed_replicate", "observed_grid", "parameter_ok",
    "neighbor_list", "short_range", "long_range", "bond", "angle", "case_dir"
]
(ROOT / "results").mkdir(exist_ok=True)
with (ROOT / "results/results.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, delimiter="\t", lineterminator="\n", fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out)
print(json.dumps({
    "total": len(out),
    "submitted": sum(bool(row["job_id"]) for row in out),
    "validated": sum(row["status"] == "validated_complete" for row in out),
    "failed": sum(row["status"] == "failed_or_invalid" for row in out),
}, indent=2))
