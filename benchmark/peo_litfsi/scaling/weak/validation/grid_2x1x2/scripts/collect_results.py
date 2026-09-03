#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
ROOT=Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/peo_weak_firstpoint_grid_2x1x2_20260819")
D=ROOT/"cases/peo_4dcu_grid_2x1x2"
row=next(csv.DictReader((ROOT/"manifest.tsv").open(),delimiter="\t"))
jobs=list(csv.DictReader((ROOT/"submitted_jobs.tsv").open(),delimiter="\t"))
q={**row,"job_id":jobs[-1]["job_id"] if jobs else "","status":"not_submitted"}
if q["job_id"]:q["status"]="queued_or_running"
meta=D/"runtime_metadata.txt"
if meta.exists():
  md=dict(x.split("=",1) for x in meta.read_text().splitlines() if "=" in x);q.update(md)
log=D/"rbmd.log"
if log.exists():
  text=log.read_text(errors="replace")
  for label,key in (("Neighbor-List","neighbor"),("Short-Range","short"),("Long-Range","long"),("Bond","bond"),("Angle","angle")):
    m=re.search(r"│\s*"+re.escape(label)+r"\s*│\s*([0-9.eE+-]+)\s*│",text)
    if m:q[key]=m.group(1)
  q["finish"]="Finish" in text
if q.get("short") and q.get("long"):q["short_plus_long"]=str(float(q["short"])+float(q["long"]))
valid=(q.get("exit_code")=="0" and q.get("finish") and int(q.get("grid_intercept_count") or 0)>0 and all(q.get(k) for k in ("neighbor","short","long","bond","angle")) and (D/"VALIDATED_COMPLETE").exists())
if valid:q["status"]="validated_complete"
elif q.get("exit_code"):q["status"]="failed_or_invalid"
fields=["job_id","name","status","exit_code","nodes","dcus","atoms","steps","processor_grid","neighbor","short","long","short_plus_long","bond","angle","elapsed_s","hostlist","grid_intercept_count","case_dir"]
with (ROOT/"results/results.tsv").open("w",newline="") as s:
  w=csv.DictWriter(s,delimiter="\t",lineterminator="\n",fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerow(q)
print(json.dumps(q,indent=2))
