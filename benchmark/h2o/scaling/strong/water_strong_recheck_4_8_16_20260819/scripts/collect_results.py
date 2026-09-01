#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path

ROOT=Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_recheck_4_8_16_20260819")
TIMING=re.compile(r"│\s*(Neighbor-List|Short-Range|Long-Range|Bond|Angle)\s*│\s*([0-9.eE+-]+)\s*│")
GRID=re.compile(r"MPI grid dimensions:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
STEP1000=re.compile(r"\|\s*1000\s*\|")
manifest=list(csv.DictReader((ROOT/"manifest.tsv").open(),delimiter="\t"))
submitted={r["case_dir"]:r["job_id"] for r in csv.DictReader((ROOT/"submitted_jobs.tsv").open(),delimiter="\t")}
out=[]
for row in manifest:
    case_dir=Path(row["case_dir"])
    result={**row,"job_id":submitted.get(row["case_dir"],""),"status":"not_submitted"}
    if result["job_id"]:
        result["status"]="queued_or_running"
    meta=case_dir/"runtime_metadata.txt"
    log=case_dir/"rbmd.log"
    if meta.exists():
        md=dict(line.split("=",1) for line in meta.read_text().splitlines() if "=" in line)
        result["exit_code"]=md.get("exit_code","")
        result["hostlist"]=md.get("hostlist","")
        result["metadata_atoms_ok"]=md.get("expected_atoms")==row["atoms"]
        result["metadata_steps_ok"]=md.get("steps")==row["steps"]
    slurm_out=next(case_dir.glob("slurm-*.out"),None)
    if slurm_out:
        slurm_text=slurm_out.read_text(errors="replace")
        match=GRID.search(slurm_text)
        result["observed_mpi_grid"]="x".join(match.groups()) if match else ""
    if log.exists():
        text=log.read_text(errors="replace")
        rep=re.search(r"replicate_sampling\s*:\s*\(([^)]+)\)",text)
        result["finish"]="Finish" in text
        result["step1000_seen"]=bool(STEP1000.search(text))
        result["observed_replicate"]=rep.group(1).replace(" ","") if rep else ""
        for key,value in TIMING.findall(text):
            result[key.lower().replace("-","_")]=value
    config=json.loads((case_dir/"run.json").read_text())
    source=json.loads((case_dir/"source_run.json").read_text())
    neighbor=config["hyper_parameters"]["neighbor"]
    coulomb=config["hyper_parameters"]["coulomb"]
    result["parameter_ok"]=(
        float(neighbor.get("cut_off"))==9
        and float(neighbor.get("r_core"))==5
        and neighbor.get("neighbor_sample_num")==30
        and coulomb.get("coulomb_sample_num")==200
        and float(coulomb.get("accuracy"))==1.0e-4
        and config["execution"].get("num_steps")==1000
        and config["hyper_parameters"]["extend"].get("fix_shake") is True
    )
    result["settings_unchanged"]=config==source
    required_timings=all(result.get(key) for key in ("neighbor_list","short_range","long_range","bond","angle"))
    if (
        result.get("exit_code")=="0"
        and result.get("finish")
        and result.get("step1000_seen")
        and result.get("observed_replicate")=="6,6,6"
        and result.get("observed_mpi_grid")==row["mpi_grid_expected"]
        and result.get("metadata_atoms_ok")
        and result.get("metadata_steps_ok")
        and result["parameter_ok"]
        and result["settings_unchanged"]
        and required_timings
    ):
        result["status"]="validated_complete"
        (case_dir/"VALIDATED_COMPLETE").touch()
    elif result.get("exit_code"):
        result["status"]="failed_or_invalid"
    out.append(result)

fields=[
    "job_id","order","name","nodes","dcus","atoms","replicate","steps","status",
    "exit_code","finish","step1000_seen","observed_replicate","observed_mpi_grid",
    "parameter_ok","settings_unchanged","metadata_atoms_ok","metadata_steps_ok",
    "neighbor_list","short_range","long_range","bond","angle","hostlist",
    "mpi_grid_expected","local_box_angstrom","case_dir"
]
with (ROOT/"results/results.tsv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,delimiter="\t",lineterminator="\n",fieldnames=fields,extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out)
print(json.dumps({
    "total":len(out),
    "submitted":sum(bool(row["job_id"]) for row in out),
    "validated":sum(row["status"]=="validated_complete" for row in out),
    "failed":sum(row["status"]=="failed_or_invalid" for row in out),
},indent=2))
