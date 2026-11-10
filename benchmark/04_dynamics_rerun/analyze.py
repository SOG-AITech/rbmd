#!/usr/bin/env python3
"""Analyze one dedicated LAMMPS observable: dielectric or small-k Szz."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from read_thermo import read

def constant(rows,key):
    values=[r[key] for r in rows]
    if not all(math.isfinite(v) and v>0 for v in values):raise ValueError(f'Invalid {key}')
    if not all(math.isclose(v,values[0],rel_tol=1e-8) for v in values):raise ValueError(f'{key} must be constant')
    return values[0]

def analyze(rows,meta,temperature=None):
    mode=meta['observable'];units=meta['units'].lower()
    if len(rows)<3:raise ValueError('Need at least three selected frames')
    if any(not math.isfinite(v) for r in rows for v in r.values()):raise ValueError('Non-finite observable')
    n=constant(rows,'Atoms')
    if n!=meta['n_atoms']:raise ValueError('Rerun/trajectory atom count mismatch')
    result=dict(observable=mode,system=meta['system'],units=units,n_atoms=n,
                first=rows[0]['Step'],last=rows[-1]['Step'],frames=len(rows),
                uncertainty='One trajectory estimate; aggregate independent trajectories, not correlated frames')
    if mode=='dielectric':
        if units!='real' or type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0:
            raise ValueError('Water requires real units and --temperature from the original constrained run')
        target=meta.get('target_temperature_K')
        if type(target) not in (int,float) or not math.isfinite(target) or target<=0:
            raise ValueError('Water requires a finite positive target_temperature_K from the original thermostat')
        if any(abs(r['c_qsum'])>1e-6 for r in rows):raise ValueError('Dielectric formula requires neutral water')
        V=constant(rows,'Volume')*1e-30
        variance=sum(statistics.pvariance([r[f'c_dip[{i}]'] for r in rows]) for i in (1,2,3))
        result.update(epsilon_r=1+variance*(1.602176634e-29)**2/(3*8.8541878128e-12*V*1.380649e-23*temperature),
            temperature_K=temperature,target_temperature_K=target,
            temperature_basis='measured mean from original constrained trajectory in the analysis window',
            dipole_variance_e2_A2=variance,
            dielectric_scope='Neutral water, consistent molecular images, fixed volume, conducting boundary')
    elif mode=='structure':
        constant(rows,'c_qnorm');points={}
        for axis in ('x','y','z'):
            L=constant(rows,'L'+axis)
            for mode_index in (1,2,3,4):
                tag=f'{axis}{mode_index}';values=[r['v_S'+tag] for r in rows]
                if min(values)<0:raise ValueError('Negative charge structure factor')
                points[tag]=dict(k=2*math.pi*mode_index/L,mean=statistics.mean(values))
        result.update(small_k_Szz=points,k_units='1/angstrom' if units=='real' else '1/sigma',
            normalization='|sum_i q_i exp(i k.r_i)|^2 / sum_i q_i^2; full-box axial reciprocal vectors; no cross-axis pooling')
    else:raise ValueError('RDF uses compare_rdf.py; choose a dedicated observable rerun')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--postprocess',type=Path,required=True,help='postprocess.json from prepare_rerun.py')
    p.add_argument('--log',type=Path,help='Defaults to lammps/log.lammps next to postprocess.json')
    p.add_argument('--first',type=int,required=True);p.add_argument('--last',type=int,required=True)
    p.add_argument('--temperature',type=float,help='Measured mean temperature from original constrained run, K; water only')
    p.add_argument('--target-temperature',type=float,help='Original thermostat target, K; required only for legacy metadata without target_temperature_K')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=json.loads(a.postprocess.read_text());log=a.log or a.postprocess.parent/'lammps/log.lammps'
    if not meta['first']<=a.first<a.last<=meta['last']:p.error('Analysis window outside processed trajectory')
    if a.target_temperature is not None:
        recorded=meta.get('target_temperature_K')
        if recorded is not None and not math.isclose(recorded,a.target_temperature,rel_tol=1e-8):
            p.error('--target-temperature conflicts with recorded thermostat target')
        meta['target_temperature_K']=a.target_temperature
    rows=[r for r in read(log) if a.first<=r['Step']<=a.last]
    if not rows or rows[0]['Step']!=a.first or rows[-1]['Step']!=a.last or any(b['Step']-x['Step']!=meta['stride'] for x,b in zip(rows,rows[1:])):
        p.error('Incomplete analysis window or missing frames')
    result=analyze(rows,meta,a.temperature)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w') as f:
        w=csv.writer(f)
        if meta['observable']=='structure':
            w.writerow(['mode','k','mean_Szz'])
            for tag,point in result['small_k_Szz'].items():w.writerow([tag,point['k'],point['mean']])
        else:
            w.writerow(rows[0].keys());w.writerows(r.values() for r in rows)
    print(a.output)

if __name__=='__main__':main()
