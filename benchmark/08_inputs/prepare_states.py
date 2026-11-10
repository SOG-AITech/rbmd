#!/usr/bin/env python3
"""Prepare five independently seeded equilibration runs; outputs remain candidates until reviewed."""
import argparse,copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from prepare import load_system,lammps_setup,thermostat,add_case,quiet,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',type=Path,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.steps<1 or not 1<=a.seed<900000000-4:p.error('Positive steps and valid seed required')
    s=load_system(a.system);base=s['rbmd']
    if a.data:base['init_configuration']['read_data']['file']=str(a.data.resolve())
    if not Path(base['init_configuration']['read_data']['file']).is_file():p.error('Input data missing')
    if a.output.exists():p.error('Output exists')
    cases=[];states=[]
    for i in range(5):
        cfg=copy.deepcopy(base);quiet(cfg,a.steps);temperature=cfg['execution']['temperature'][0]
        text=lammps_setup(s,cfg,1e-6)+thermostat(s,cfg)
        text+=f'velocity all create {temperature} {a.seed+i} mom yes rot yes dist gaussian\n'
        text+='thermo_style custom step time atoms temp pe press density\nthermo_modify norm no lost error format float %.17g\nthermo 1000\n'
        text+=f'run {a.steps}\nwrite_data state.data\nwrite_restart state.restart\n'
        case=add_case(a.output,f'state{i+1}',s,cfg,'lammps',text,purpose='equilibration_candidate')
        cases.append(str(case.resolve()));states.append(str((case/'state.data').resolve()))
    (a.output/'cases.txt').write_text('\n'.join(cases)+'\n')
    write_json(a.output/'states.json',dict(states=states,seeds=list(range(a.seed,a.seed+5)),
        status='CANDIDATES_NOT_RUN',independence_verified=False,equilibration_verified=False,
        review='Check density/energy/stress stationarity, relaxation and replica decorrelation before using --states'))
    print(a.output)
if __name__=='__main__':main()
