#!/usr/bin/env python3
"""Five independent run summaries -> mean, SD, SEM and Student-t 95% interval."""
import argparse
import json
import math
from pathlib import Path
import statistics

def summarize(values):
    if len(values)!=5 or not all(math.isfinite(v) for v in values):raise ValueError('Exactly five finite independent run estimates required')
    avg=statistics.mean(values);sd=statistics.stdev(values);sem=sd/math.sqrt(5)
    return dict(n=5,mean=avg,sd=sd,sem=sem,ci95=[avg-2.7764451052*sem,avg+2.7764451052*sem])

def aggregate(rows,temperature_tolerance=1.0):
    if len(rows)!=5:raise ValueError('Exactly five independent trajectories required')
    first=rows[0]
    for key in ('observable','system','units','n_atoms'):
        if any(r[key]!=first[key] for r in rows):raise ValueError(f'Mismatched {key}')
    mode=first['observable'];result={k:first[k] for k in ('observable','system','units','n_atoms')}
    if mode=='dielectric':
        temperatures=[r.get('temperature_K') for r in rows]
        if any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in temperatures):
            raise ValueError('Every dielectric result requires a finite positive temperature_K')
        targets=[r.get('target_temperature_K') for r in rows]
        if any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in targets):
            raise ValueError('Every dielectric result requires a finite positive target_temperature_K; reanalyze legacy results')
        if any(not math.isclose(t,targets[0],rel_tol=1e-8,abs_tol=0) for t in targets):
            raise ValueError('Mismatched target_temperature_K; do not pool different thermostat targets')
        if type(temperature_tolerance) not in (int,float) or not math.isfinite(temperature_tolerance) or temperature_tolerance<0:
            raise ValueError('temperature tolerance must be finite and nonnegative')
        if any(abs(t-targets[0])>temperature_tolerance for t in temperatures):
            raise ValueError('Measured temperature_K exceeds the allowed deviation from target_temperature_K')
        result.update(target_temperature_K=targets[0],measured_temperatures_K=temperatures,
            measured_temperature_summary_K=summarize(temperatures),
            temperature_tolerance_K=temperature_tolerance,
            temperature_policy='Each epsilon_r uses its own measured temperature; pooled only at a common thermostat target, without rescaling. Tolerance is an operational check, not a convergence certificate.')
        result['epsilon_r']=summarize([r['epsilon_r'] for r in rows])
    elif mode=='structure':
        points=first['small_k_Szz'];result['small_k_Szz']={};result['k_units']=first['k_units']
        for r in rows:
            if r['k_units']!=first['k_units'] or r['small_k_Szz'].keys()!=points.keys():raise ValueError('Mismatched k modes/units')
        for tag,point in points.items():
            if any(not math.isclose(r['small_k_Szz'][tag]['k'],point['k'],rel_tol=1e-8) for r in rows):
                raise ValueError('Different k values; do not pool different boxes')
            result['small_k_Szz'][tag]=dict(k=point['k'],**summarize([r['small_k_Szz'][tag]['mean'] for r in rows]))
    else:raise ValueError('Unsupported observable')
    result['interpretation']='CI describes independent-run estimates; correlated frames are not independent repeats'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',nargs=5,type=Path)
    p.add_argument('--independent-states',action='store_true',required=True,help='Confirm separately equilibrated starts, not five copies of one deterministic run')
    p.add_argument('--temperature-tolerance',type=float,default=1.0,help='Maximum absolute deviation of each measured mean from the common thermostat target, K (default 1); dielectric only')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if len(set(p.resolve() for p in a.results))!=5:raise ValueError('Duplicate result paths')
    rows=[json.loads(p.read_text()) for p in a.results]
    result=aggregate(rows,a.temperature_tolerance)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.output)

if __name__=='__main__':main()
