#!/usr/bin/env python3
"""Prepare only. No scheduler commands are executed."""
import argparse,copy,json,math,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
from prepare import add_case,lammps_setup,thermostat,quiet

def main():
 p=argparse.ArgumentParser();p.add_argument('--system',choices=['water','lj','all'],default='all');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve()
 if out.exists():p.error('Use a new output directory')
 matrix=json.loads((Path(__file__).parent/'matrix.json').read_text());jobs=[]
 for name,system in matrix['systems'].items():
  if a.system not in ['all',name]:continue
  template=json.loads((ROOT/'systems'/f'{name}.json').read_text())
  for g in system['groups']:
   assert g['ranks']==4*g['nodes']==math.prod(g['grid'])
   assert g['atoms']==(72981 if name=='water' else 200000)*math.prod(g['config']['init_configuration']['read_data']['replicate'])
   for rep in range(1,system['repeats']+1):
    for engine in ['rbmd','lammps']:
     cfg=copy.deepcopy(g['config']);quiet(cfg,matrix['steps'],True)
     text=None
     if engine=='lammps':
      text=lammps_setup(template,cfg,1e-4).replace('boundary p p p','boundary p p p\nprocessors '+' '.join(map(str,g['grid'])))+thermostat(template,cfg)+f'thermo_style custom step atoms temp\nthermo {matrix["steps"]}\ntimer full nosync\nrun {matrix["steps"]}\n'
     dest=add_case(out,f'{name}/{g["id"]}/repeat{rep}/{engine}',template,cfg,engine,text,measured=matrix['steps'],sync='on',purpose='05-native-sync')
     meta=json.loads((dest/'case.json').read_text());meta.update(N=g['atoms'],mpi_grid=g['grid'],layout={'nodes':g['nodes'],'gpus':g['ranks'],'gpus_per_node':4},binary=system['binary'] if engine=='rbmd' else '/public/home/acymkxdx47/software/rbm26/build-hip-single/lmp',binary_sha256=system['sha256'] if engine=='rbmd' else '8689898e99e82fca4d9262c1837ef66dbe3f3dae2db4adb974b87efd7c0bc306',preload=system.get('preload') if engine=='rbmd' else None,preload_sha256=system.get('preload_sha256'),force_evaluations=matrix['steps']+1 if engine=='rbmd' else matrix['steps'])
     (dest/'case.json').write_text(json.dumps(meta,indent=2))
     (dest/'run.sh').write_text('#!/bin/bash\nset -euo pipefail\n'+shlex.join(['bash',str(ROOT/'05_timing_scaling/run_case.sh'),engine,str(dest)])+'\n')
     job=dest/'job.slurm';job.write_text(f'#!/bin/bash\n#SBATCH --partition=hx1hdnormal01\n#SBATCH --nodes={g["nodes"]}\n#SBATCH --ntasks-per-node=4\n#SBATCH --cpus-per-task=4\n#SBATCH --gres=dcu:4\n#SBATCH --time=02:00:00\nset -euo pipefail\nexec bash '+shlex.quote(str(dest/'run.sh'))+'\n');jobs.append(str(job))
 (out/'jobs.txt').write_text('\n'.join(jobs)+'\n');print(f'Prepared {len(jobs)} jobs; NOT submitted')
if __name__=='__main__':main()
