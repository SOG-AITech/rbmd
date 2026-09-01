#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
ROOT=Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_linear_same_sizes_shape_control_20260818")
timing=re.compile(r"│\s*(Neighbor-List|Short-Range|Long-Range|Bond|Angle)\s*│\s*([0-9.eE+-]+)\s*│")
manifest=list(csv.DictReader((ROOT/"manifest.tsv").open(),delimiter="\t"))
submitted={r["case_dir"]:r["job_id"] for r in csv.DictReader((ROOT/"submitted_jobs.tsv").open(),delimiter="\t")}
out=[]
for row in manifest:
 d=Path(row["case_dir"]); q={**row,"job_id":submitted.get(row["case_dir"],""),"status":"not_submitted"}
 if q["job_id"]: q["status"]="queued_or_running"
 meta=d/"runtime_metadata.txt"; log=d/"rbmd.log"
 if meta.exists(): q["exit_code"]=dict(x.split("=",1) for x in meta.read_text().splitlines() if "=" in x).get("exit_code","")
 if log.exists():
  text=log.read_text(errors="replace"); rep=re.search(r"replicate_sampling\s*:\s*\(([^)]+)\)",text)
  q["finish"]="Finish" in text; q["observed_replicate"]=rep.group(1).replace(" ","") if rep else ""
  for k,v in timing.findall(text): q[k.lower().replace("-","_")]=v
 cfg=json.loads((d/"run.json").read_text()); n=cfg["hyper_parameters"]["neighbor"]; c=cfg["hyper_parameters"]["coulomb"]
 q["parameter_ok"]=(n.get("cut_off")==9 and n.get("r_core")==5 and n.get("neighbor_sample_num")==30 and c.get("coulomb_sample_num")==200 and cfg["execution"].get("num_steps")==1000)
 if q.get("exit_code")=="0" and q.get("finish") and q.get("observed_replicate")==row["replicate"].replace("x",",") and q.get("short_range") and q.get("long_range") and q["parameter_ok"]:
  q["status"]="validated_complete"; (d/"VALIDATED_COMPLETE").touch()
 elif q.get("exit_code"): q["status"]="failed_or_invalid"
 out.append(q)
fields=["job_id","name","atoms","replicate","dcus","steps","mpi_grid","local_shape_units","status","exit_code","finish","observed_replicate","parameter_ok","neighbor_list","short_range","long_range","bond","angle","case_dir"]
with (ROOT/"results/results.tsv").open("w",newline="") as f:
 w=csv.DictWriter(f,delimiter="\t",lineterminator="\n",fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(out)
print(json.dumps({"total":len(out),"submitted":sum(bool(x["job_id"]) for x in out),"validated":sum(x["status"]=="validated_complete" for x in out),"failed":sum(x["status"]=="failed_or_invalid" for x in out)},indent=2))
