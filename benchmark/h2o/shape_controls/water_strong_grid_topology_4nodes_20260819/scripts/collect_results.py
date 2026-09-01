#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
ROOT=Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819")
rbmd_timing=lambda name: re.compile(rf"│\s*{re.escape(name)}\s*│\s*([0-9.eE+-]+)\s*│")
loop_re=re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms")
section=lambda name: re.compile(rf"^{name}\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)",re.M)
rows=list(csv.DictReader((ROOT/"manifest.tsv").open(),delimiter="\t"))
submitted={r["case_dir"]:r["job_id"] for r in csv.DictReader((ROOT/"submitted_jobs.tsv").open(),delimiter="\t")}
out=[]
for row in rows:
  d=Path(row["case_dir"]); q={**row,"job_id":submitted.get(row["case_dir"],""),"status":"not_submitted"}
  if q["job_id"]: q["status"]="queued_or_running"
  meta=d/"runtime_metadata.txt"
  if meta.exists():
    md=dict(x.split("=",1) for x in meta.read_text().splitlines() if "=" in x)
    for k in ("exit_code","elapsed_s","hostlist","actual_atoms","actual_steps","preload_sha256","grid_intercept_count"): q[k]=md.get(k,"")
  if row["method"]=="RBMD":
    log=d/"rbmd.log"
    if log.exists():
      text=log.read_text(errors="replace")
      for label,key in (("Neighbor-List","neighbor"),("Short-Range","short"),("Long-Range","long"),("Bond","bond"),("Angle","angle")):
        m=rbmd_timing(label).search(text)
        if m:q[key]=m.group(1)
      q["finish"]="Finish" in text
      q["grid_ok"]=int(q.get("grid_intercept_count") or 0)>0
    if q.get("short") and q.get("long"):q["short_plus_long"]=str(float(q["short"])+float(q["long"]))
    valid=(q.get("finish") and q.get("grid_ok") and q.get("neighbor") and
           q.get("short") and q.get("long") and q.get("bond") and q.get("angle"))
  else:
    log=d/"log.lammps"
    if log.exists():
      text=log.read_text(errors="replace"); m=loop_re.search(text)
      if m:q.update({"loop_time":m.group(1),"loop_steps":m.group(3),"loop_atoms":m.group(4)})
      for label,key in (("Pair","pair"),("Kspace","kspace")):
        m=section(label).search(text)
        if m:q[key+"_avg"]=m.group(2);q[key+"_max"]=m.group(3)
      if q.get("pair_avg") and q.get("kspace_avg"):q["short_plus_long"]=str((float(q["pair_avg"])+float(q["kspace_avg"]))/1000.0)
    valid=q.get("loop_atoms")==row["atoms"] and q.get("loop_steps")==row["steps"] and q.get("pair_avg") and q.get("pair_max") and q.get("kspace_avg") and q.get("kspace_max")
  if q.get("exit_code")=="0" and (d/"VALIDATED_COMPLETE").exists() and valid:q["status"]="validated_complete"
  elif q.get("exit_code"):q["status"]="failed_or_invalid"
  out.append(q)
fields=["job_id","name","method","grid_label","processor_grid","status","exit_code","nodes","dcus","atoms","steps","short","long","pair_avg","pair_max","kspace_avg","kspace_max","short_plus_long","neighbor","bond","angle","loop_time","elapsed_s","hostlist","preload_sha256","grid_intercept_count","case_dir"]
with (ROOT/"results/results.tsv").open("w",newline="") as s:
  w=csv.DictWriter(s,delimiter="\t",lineterminator="\n",fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(out)
print(json.dumps({"total":len(out),"validated":sum(x["status"]=="validated_complete" for x in out),"failed":sum(x["status"]=="failed_or_invalid" for x in out)},indent=2))
