#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
ROOT=Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/peo_weak_firstpoint_repeat_20260819")
timing=lambda name: re.compile(rf"│\s*{re.escape(name)}\s*│\s*([0-9.eE+-]+)\s*│")
manifest=list(csv.DictReader((ROOT/"manifest.tsv").open(),delimiter="\t"))
submitted={r["case_dir"]:r["job_id"] for r in csv.DictReader((ROOT/"submitted_jobs.tsv").open(),delimiter="\t")}
out=[]
for row in manifest:
    d=Path(row["case_dir"]); q={**row,"job_id":submitted.get(row["case_dir"],""),"status":"not_submitted"}
    meta=d/"runtime_metadata.txt"; log=d/"rbmd.log"
    if q["job_id"]: q["status"]="queued_or_running"
    if meta.exists():
        md=dict(x.split("=",1) for x in meta.read_text().splitlines() if "=" in x)
        q.update({"exit_code":md.get("exit_code",""),"elapsed_s":md.get("elapsed_s",""),"hostlist":md.get("hostlist","")})
    if log.exists():
        text=log.read_text(errors="replace")
        for label,key in (("Neighbor-List","neighbor"),("Short-Range","short"),("Long-Range","long"),("Bond","bond"),("Angle","angle")):
            m=timing(label).search(text)
            if m: q[key]=m.group(1)
        q["finish"]="Finish" in text
    q["short_plus_long"]=""
    if q.get("short") and q.get("long"):
        q["short_plus_long"]=str(float(q["short"])+float(q["long"]))
    if q.get("exit_code")=="0" and (d/"VALIDATED_COMPLETE").exists() and q.get("finish") and q.get("short") and q.get("long"):
        q["status"]="validated_complete"
    elif q.get("exit_code"):
        q["status"]="failed_or_invalid"
    out.append(q)
fields=["job_id","name","status","exit_code","nodes","dcus","atoms","steps","short","long","short_plus_long","neighbor","bond","angle","elapsed_s","hostlist","case_dir"]
with (ROOT/"results/results.tsv").open("w",newline="") as stream:
    w=csv.DictWriter(stream,delimiter="\t",lineterminator="\n",fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(out)
print(json.dumps({"total":len(out),"validated":sum(x["status"]=="validated_complete" for x in out),"failed":sum(x["status"]=="failed_or_invalid" for x in out)},indent=2))
