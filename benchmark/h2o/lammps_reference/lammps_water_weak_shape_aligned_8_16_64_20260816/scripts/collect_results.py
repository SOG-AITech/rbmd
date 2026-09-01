#!/usr/bin/env python3
import csv, json, re
from pathlib import Path

ROOT=Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/lammps_water_weak_shape_aligned_8_16_64_20260816")
loop_re=re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms")
section_re=lambda name: re.compile(rf"^{name}\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)",re.M)
manifest=list(csv.DictReader((ROOT/"manifest.tsv").open(),delimiter="\t"))
submitted={r["case_dir"]:r["job_id"] for r in csv.DictReader((ROOT/"submitted_jobs.tsv").open(),delimiter="\t")}
out=[]
for row in manifest:
    d=Path(row["case_dir"])
    q={**row,"job_id":submitted.get(row["case_dir"],""),"status":"not_submitted"}
    if q["job_id"]:
        q["status"]="queued_or_running"
    meta=d/"runtime_metadata.txt"
    log=d/"log.lammps"
    if meta.exists():
        md=dict(x.split("=",1) for x in meta.read_text().splitlines() if "=" in x)
        q["exit_code"]=md.get("exit_code","")
        q["metadata_atoms_ok"]=md.get("actual_atoms")==row["atoms"]
        q["metadata_steps_ok"]=md.get("actual_steps")==row["steps"]
        q["metadata_processors_ok"]=md.get("processor_grid")==row["processor_grid"]
    if log.exists():
        text=log.read_text(errors="replace")
        loop=loop_re.search(text)
        if loop:
            q["loop_time"]=loop.group(1)
            q["loop_procs"]=loop.group(2)
            q["loop_steps"]=loop.group(3)
            q["loop_atoms"]=loop.group(4)
        for name in ("Pair","Kspace"):
            match=section_re(name).search(text)
            if match:
                key=name.lower()
                q[f"{key}_min"]=match.group(1)
                q[f"{key}_avg"]=match.group(2)
                q[f"{key}_max"]=match.group(3)
                q[f"{key}_avg_per_step"]=str(float(match.group(2))/1000.0)
                q[f"{key}_max_per_step"]=str(float(match.group(3))/1000.0)
        q["parameter_ok"]=(
            "pair_style      lj/cut/coul/long 9.0 9.0" in text
            and "kspace_style    pppm 1.0e-4" in text
            and "timer           full sync" in text
            and "run             1000" in text
        )
        processor_pattern = (
            r"^processors\s+"
            + row["processor_grid"].replace("x", r"\s+")
            + r"\s*$"
        )
        q["processor_command_ok"]=bool(re.search(processor_pattern, text, re.M))
    marker=(d/"VALIDATED_COMPLETE").exists()
    if (
        q.get("exit_code")=="0"
        and marker
        and q.get("metadata_atoms_ok")
        and q.get("metadata_steps_ok")
        and q.get("metadata_processors_ok")
        and q.get("loop_atoms")==row["atoms"]
        and q.get("loop_steps")==row["steps"]
        and q.get("pair_avg") and q.get("pair_max")
        and q.get("kspace_avg") and q.get("kspace_max")
        and q.get("parameter_ok")
        and q.get("processor_command_ok")
    ):
        q["status"]="validated_complete"
    elif q.get("exit_code"):
        q["status"]="failed_or_invalid"
    out.append(q)

fields=["job_id","name","replicate","atoms","nodes","dcus","dcus_per_node","steps","status","exit_code","metadata_atoms_ok","metadata_steps_ok","metadata_processors_ok","loop_time","loop_procs","loop_steps","loop_atoms","pair_min","pair_avg","pair_max","pair_avg_per_step","pair_max_per_step","kspace_min","kspace_avg","kspace_max","kspace_avg_per_step","kspace_max_per_step","parameter_ok","processor_command_ok","processor_grid","mpi_grid_expected","local_replicate_expected","case_dir"]
with (ROOT/"results/results.tsv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,delimiter="\t",lineterminator="\n",fieldnames=fields,extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out)
print(json.dumps({
    "total":len(out),
    "submitted":sum(bool(x["job_id"]) for x in out),
    "validated":sum(x["status"]=="validated_complete" for x in out),
    "failed":sum(x["status"]=="failed_or_invalid" for x in out),
},indent=2))
