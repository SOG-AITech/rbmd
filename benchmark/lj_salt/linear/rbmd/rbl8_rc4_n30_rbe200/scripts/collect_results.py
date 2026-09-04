#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
ROOT=Path("/public/home/acymkxdx47/rbmd/build/new_gyf_test/rbmd_linear_subnode_20260815/corrected_ljsalt_RBE200_RBL30")
timing=re.compile(r"│\s*(Neighbor-List|Short-Range|Long-Range)\s*│\s*([0-9.eE+-]+)\s*│")
manifest=list(csv.DictReader((ROOT/"manifest.tsv").open(),delimiter="\t"))
submitted={r["case_dir"]:r["job_id"] for r in csv.DictReader((ROOT/"submitted_jobs.tsv").open(),delimiter="\t")}
out=[]
for r in manifest:
 d=Path(r["case_dir"]); q={**r,"job_id":submitted.get(r["case_dir"],""),"status":"pending"}
 meta=d/"runtime_metadata.txt"; log=d/"rbmd.log"; run=d/"run.json"
 if meta.exists(): q.update({"exit_code":dict(x.split("=",1) for x in meta.read_text().splitlines() if "=" in x).get("exit_code","")})
 if log.exists() and run.exists():
  txt=log.read_text(errors="replace"); cfg=json.loads(run.read_text()); rep=re.search(r"replicate_sampling\s*:\s*\(([^)]+)\)",txt)
  q["finish"]="Finish" in txt; q["observed_replicate"]=rep.group(1).replace(" ","") if rep else ""
  for k,v in timing.findall(txt): q[k.lower().replace("-","_")]=v
  param_ok=(cfg["hyper_parameters"]["neighbor"]["neighbor_sample_num"]==30 and cfg["hyper_parameters"]["coulomb"]["coulomb_sample_num"]==200 and cfg["hyper_parameters"]["neighbor"]["cut_off"]==8 and cfg["hyper_parameters"]["neighbor"]["r_core"]==4)
  q["parameter_ok"]=param_ok
  if q.get("exit_code")=="0" and q["finish"] and q["observed_replicate"]==r["rep"].replace("x",",") and q.get("short_range") and q.get("long_range") and param_ok:
   q["status"]="validated_complete"; (d/"VALIDATED_COMPLETE").touch()
  elif q.get("exit_code"): q["status"]="failed_or_invalid"
 out.append(q)
fields=["job_id","system","mode","name","gpus","rep","atoms","steps","status","exit_code","finish","observed_replicate","parameter_ok","neighbor_list","short_range","long_range","case_dir"]
with (ROOT/"results"/"results.tsv").open("w",newline="") as f:
 w=csv.DictWriter(f,delimiter="\t",lineterminator="\n",fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(out)
print(json.dumps({"total":len(out),"validated":sum(x["status"]=="validated_complete" for x in out),"rows":out},indent=2))
