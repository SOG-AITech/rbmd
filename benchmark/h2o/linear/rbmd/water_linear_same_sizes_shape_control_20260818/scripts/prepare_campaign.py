#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


ROOT = Path(
    "/public/home/acymkxdx47/rbmd/build/new_gyf_test/"
    "water_linear_same_sizes_shape_control_20260818"
)
SOURCE = Path(
    "/public/home/acymkxdx47/rbmd/build/new_gyf_test/"
    "water_linear_validation_20260815/cases/atoms_01970487/run.json"
)
RBMD_BIN = Path("/public/home/acymkxdx47/rbmd/build/rbmd")
CASES = [
    ("atoms_00583848", 583_848, [2, 2, 2]),
    ("atoms_01313658", 1_313_658, [3, 3, 2]),
    ("atoms_01970487", 1_970_487, [3, 3, 3]),
    ("atoms_03940974", 3_940_974, [3, 6, 3]),
    ("atoms_07881948", 7_881_948, [6, 6, 3]),
    ("atoms_15763896", 15_763_896, [6, 6, 6]),
]


def atom_count(path: Path) -> int:
    with path.open("r", errors="replace") as stream:
        for _ in range(160):
            match = re.match(r"^\s*(\d+)\s+atoms\s*$", stream.readline())
            if match:
                return int(match.group(1))
    raise RuntimeError(f"Atom count not found in {path}")


def slurm_script(name: str, case_dir: Path) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=wat_shape_{name[-8:]}
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4
RBMD_BIN={RBMD_BIN}
ROCM_PATH=/public/software/compiler/dtk-25.04.4
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${{LD_LIBRARY_PATH:-}}"
export UCX_TLS="${{UCX_TLS:-^cma,rocm_ipc}}"
export UCX_MEMTYPE_CACHE="${{UCX_MEMTYPE_CACHE:-n}}"
export RBMD_DEBUG_NEIGHBOR_SKIN=0
export OMP_NUM_THREADS=1
ulimit -c 0
cd "${{SLURM_SUBMIT_DIR}}"
test "$(pwd -P)" = "$(readlink -f {case_dir})"
test -x "$RBMD_BIN"
test -s ./run.json

start_epoch=$(date +%s)
printf 'job_id=%s\nnodes=%s\ntasks=%s\ndcus_per_node=4\nstart_epoch=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$start_epoch" \
  > runtime_metadata.txt
set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN -x OMP_NUM_THREADS \
  "$RBMD_BIN" -j ./run.json
rc=$?
set -e
end_epoch=$(date +%s)
printf 'end_epoch=%s\nelapsed_s=%s\nexit_code=%s\n' \
  "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
exit "$rc"
"""


if ROOT.exists():
    raise RuntimeError(f"Refusing to overwrite existing campaign: {ROOT}")
if not SOURCE.is_file() or not RBMD_BIN.is_file():
    raise RuntimeError("Missing source run.json or RBMD binary")

config_template = json.loads(SOURCE.read_text())
neighbor = config_template["hyper_parameters"]["neighbor"]
coulomb = config_template["hyper_parameters"]["coulomb"]
if not (
    neighbor.get("cut_off") == 9
    and neighbor.get("r_core") == 5
    and neighbor.get("neighbor_sample_num") == 30
    and coulomb.get("coulomb_sample_num") == 200
    and config_template["execution"].get("num_steps") == 1000
):
    raise RuntimeError("Unexpected Water parameters in source run.json")

data_path = Path(config_template["init_configuration"]["read_data"]["file"])
base_atoms = atom_count(data_path)
if base_atoms != 72_981:
    raise RuntimeError(f"Unexpected base atom count: {base_atoms}")

for folder in ("cases", "scripts", "logs", "results", "archive"):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

rows = []
for name, atoms, replicate in CASES:
    config = json.loads(json.dumps(config_template))
    config["init_configuration"]["read_data"]["replicate"] = replicate
    if base_atoms * replicate[0] * replicate[1] * replicate[2] != atoms:
        raise RuntimeError(f"Atom mismatch for {name}")
    case_dir = ROOT / "cases" / name
    case_dir.mkdir()
    (case_dir / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    (case_dir / "run.slurm").write_text(slurm_script(name, case_dir))
    (case_dir / "run.slurm").chmod(0o750)
    row = {
        "name": name,
        "atoms": atoms,
        "replicate": "x".join(map(str, replicate)),
        "nodes": 1,
        "dcus": 4,
        "steps": 1000,
        "cut_off": 9,
        "r_core": 5,
        "rbl_sample": 30,
        "rbe_sample": 200,
        "mpi_grid": "2x2x1",
        "local_shape_units": "x".join(
            f"{value:g}" for value in (replicate[0] / 2, replicate[1] / 2, replicate[2])
        ),
        "case_dir": str(case_dir),
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(row, indent=2) + "\n")
    rows.append(row)

with (ROOT / "manifest.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
(ROOT / "submitted_jobs.tsv").write_text(
    "job_id\tname\tcase_dir\tpartition\tsubmitted_at\n"
)
(ROOT / "campaign.json").write_text(
    json.dumps(
        {
            "purpose": "Water fixed-4-DCU linear-complexity same-size shape controls",
            "new_cases": len(rows),
            "mpi_grid": "2x2x1 from MPI_Dims_create",
            "shape_policy": "Preserve all six atom counts; keep local aspect ratio <= 2 and avoid the prior long-x 3.94M orientation.",
            "parameters": {"cut_off": 9, "r_core": 5, "RBL": 30, "RBE": 200, "steps": 1000},
            "policy": "Free partition, one manifest-recorded job active at a time, no exclusive.",
        },
        indent=2,
    )
    + "\n"
)

submitter = r'''#!/bin/bash
set -u
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_linear_same_sizes_shape_control_20260818
MANIFEST="$ROOT/manifest.tsv"
SUBMITTED="$ROOT/submitted_jobs.tsv"
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75
test "$(grep -R -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include=run.slurm | awk -F: '{s+=$2} END{print s+0}')" = 0 || exit 2
while true; do
  latest=$(awk -F'\t' 'NR>1{id=$1} END{print id}' "$SUBMITTED")
  if [[ -n "$latest" ]] && squeue -h -j "$latest" | grep -q .; then sleep 60; continue; fi
  case_dir=$(awk -F'\t' 'NR==FNR{if(FNR>1)seen[$3]=1;next} FNR>1&&!seen[$13]{print $13;exit}' "$SUBMITTED" "$MANIFEST")
  [[ -n "$case_dir" ]] || { echo "all same-size shape controls left the queue"; exit 0; }
  name=$(basename "$case_dir")
  if job=$(cd "$case_dir" && sbatch --parsable ./run.slurm); then
    printf '%s\t%s\t%s\t%s\t%s\n' "$job" "$name" "$case_dir" hx1hdnormal "$(date -Iseconds)" >> "$SUBMITTED"
    echo "submitted $job $name"
  else
    echo "submission deferred for $name" >&2
  fi
  sleep 60
done
'''
(ROOT / "scripts/submit_remaining.sh").write_text(submitter)
(ROOT / "scripts/submit_remaining.sh").chmod(0o750)

collector = r'''#!/usr/bin/env python3
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
'''
(ROOT / "scripts/collect_results.py").write_text(collector)
(ROOT / "scripts/collect_results.py").chmod(0o750)

archive = r'''#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_linear_same_sizes_shape_control_20260818
OUT="$ROOT/archive/water_linear_same_sizes_shape_control_$(date +%Y%m%dT%H%M%S).tgz"
tar -czf "$OUT" --exclude='./archive' -C "$ROOT" .
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
'''
(ROOT / "scripts/archive_campaign.sh").write_text(archive)
(ROOT / "scripts/archive_campaign.sh").chmod(0o750)

print(json.dumps({"root": str(ROOT), "new_cases": rows, "active_exclusive": 0}, indent=2))
