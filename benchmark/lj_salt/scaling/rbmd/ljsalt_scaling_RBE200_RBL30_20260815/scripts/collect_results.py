#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path

ROOT = Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/ljsalt_scaling_RBE200_RBL30_20260815")
timing_re = re.compile(r"│\s*(Neighbor-List|Short-Range|Long-Range|Bond|Angle)\s*│\s*([0-9.eE+-]+)\s*│")
manifest = list(csv.DictReader((ROOT / "manifest.tsv").open(), delimiter="\t"))
submitted = {
    row["case_dir"]: row["job_id"]
    for row in csv.DictReader((ROOT / "submitted_jobs.tsv").open(), delimiter="\t")
}
rows = []
for row in manifest:
    case_dir = Path(row["case_dir"])
    result = {**row, "job_id": submitted.get(row["case_dir"], ""), "status": "not_submitted"}
    metadata_path = case_dir / "runtime_metadata.txt"
    log_path = case_dir / "rbmd.log"
    if result["job_id"]:
        result["status"] = "queued_or_running"
    if metadata_path.exists():
        runtime = dict(
            line.split("=", 1)
            for line in metadata_path.read_text().splitlines()
            if "=" in line
        )
        result["exit_code"] = runtime.get("exit_code", "")
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        replicate_match = re.search(r"replicate_sampling\s*:\s*\(([^)]+)\)", text)
        result["finish"] = "Finish" in text
        result["observed_replicate"] = (
            replicate_match.group(1).replace(" ", "") if replicate_match else ""
        )
        for name, value in timing_re.findall(text):
            result[name.lower().replace("-", "_")] = value
    config = json.loads((case_dir / "run.json").read_text())
    source = json.loads((case_dir / "source_run.json").read_text())
    neighbor = config["hyper_parameters"]["neighbor"]
    coulomb = config["hyper_parameters"]["coulomb"]
    source["hyper_parameters"]["neighbor"]["neighbor_sample_num"] = 30
    source["hyper_parameters"]["coulomb"]["coulomb_sample_num"] = 200
    result["parameter_ok"] = (
        neighbor.get("cut_off") == 8.0
        and neighbor.get("r_core") == 4.0
        and neighbor.get("neighbor_sample_num") == 30
        and coulomb.get("coulomb_sample_num") == 200
    )
    result["other_settings_unchanged"] = config == source
    if (
        result.get("exit_code") == "0"
        and result.get("finish")
        and result.get("observed_replicate") == row["replicate"].replace("x", ",")
        and result.get("short_range")
        and result.get("long_range")
        and result["parameter_ok"]
        and result["other_settings_unchanged"]
    ):
        result["status"] = "validated_complete"
        (case_dir / "VALIDATED_COMPLETE").touch()
    elif result.get("exit_code"):
        result["status"] = "failed_or_invalid"
    rows.append(result)

fields = [
    "job_id", "mode", "case", "nodes", "total_dcus", "replicate", "target_atoms",
    "steps", "status", "exit_code", "finish", "observed_replicate", "parameter_ok",
    "other_settings_unchanged", "neighbor_list", "short_range", "long_range", "bond",
    "angle", "case_dir"
]
with (ROOT / "results" / "results.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, delimiter="\t", lineterminator="\n", fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({
    "total": len(rows),
    "submitted": sum(bool(row["job_id"]) for row in rows),
    "validated": sum(row["status"] == "validated_complete" for row in rows),
    "failed_or_invalid": sum(row["status"] == "failed_or_invalid" for row in rows),
}, indent=2))
