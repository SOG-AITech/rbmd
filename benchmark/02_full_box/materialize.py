#!/usr/bin/env python3
"""Prepare LAMMPS replication and matching RBMD replicated/explicit input pair."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from prepare import load_system,lammps_setup,add_case,write_json,quiet

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--system',type=Path,required=True)
    p.add_argument('--replicate',type=int,nargs=3,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=1000);a=p.parse_args()
    if min(a.replicate)<1 or a.samples<2:p.error('Positive replication, samples >=2 required')
    s=load_system(a.system);cfg=s['rbmd'];a.output.mkdir(parents=True,exist_ok=False)
    setup=lammps_setup(s,cfg,1e-6)
    # Rebuild PPPM after LAMMPS expands the periodic cell.
    text=setup+'replicate '+' '.join(map(str,a.replicate))+(' bond/periodic' if s['bonded_styles'] else '')+'\nrun 0\nwrite_data expanded.data\nwrite_dump all custom reference.dump id x y z fx fy fz modify sort id format float %.17g\n'
    material_cfg=copy.deepcopy(cfg)
    material_cfg['init_configuration']['read_data']['replicate']=a.replicate
    material=add_case(a.output,'materialize',s,material_cfg,'lammps',text,lammps_role='reference')
    quiet(cfg,0);cfg['execution']['diagnostics'].update(fixed_samples=a.samples,forces=True,interval=1)
    cfg['hyper_parameters']['coulomb']['random_seed']=20261016
    for name in ('replicated','explicit'):
        c=copy.deepcopy(cfg)
        if name=='replicated':c['init_configuration']['read_data']['replicate']=a.replicate
        else:c['init_configuration']['read_data']['file']=str((material/'expanded.data').resolve())
        add_case(a.output,name,s,c,'rbmd')
    write_json(a.output/'protocol.json',dict(replicate=a.replicate,identical_replicas_only=True,
        next_test='Independently equilibrate the FULL expanded box, then run force preparation using that materialized data. Initial identical replicas cannot prove decorrelated-box correctness.'))
    print(a.output)

if __name__=='__main__':main()
