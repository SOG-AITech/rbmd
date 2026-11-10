#!/usr/bin/env python3
"""Summarize all completed cutoff timings against one fixed RBMD baseline."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_timing_scaling'))
from collect import timing

def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['experiment']!='tuning':raise ValueError('Expected a tuning manifest')
    paths=[Path(p) for p in (root/'cases.txt').read_text().splitlines()]
    if len(paths)!=manifest['cases']:raise ValueError('Incomplete case list')
    rows=[timing(p) for p in paths]
    if len({(r['nodes'],r['gpus']) for r in rows})!=1:raise ValueError('Mixed layouts; use a separate root for each GPU count')
    if any(r['launch_blocking']!='off' for r in rows):raise ValueError('Use production timings')
    audits=[json.loads((p/'preflight.json').read_text()) for p in paths]
    if any(r['status']!='PASS' for r in audits):raise ValueError('Preflight failed')
    if len({r['data_sha256'] for r in audits})!=1:raise ValueError('Different physical inputs')
    baseline=[r['ms_per_step'] for r in rows if r['engine']=='rbmd']
    if not baseline:raise ValueError('Missing RBMD baseline')
    rbmd=statistics.median(baseline); groups={}
    for r in rows:
        if r['engine']=='lammps':groups.setdefault(r['lammps_coulomb_cutoff'],[]).append(r)
    if not groups:raise ValueError('Missing PPPM timings')
    points=[]
    for cutoff,values in sorted(groups.items()):
        if len(values)!=len(baseline):raise ValueError('Incomplete repeats')
        for v in values:
            if not all(k in v['solver_settings'] for k in ('pppm_grid','pppm_order','gewald')):
                raise ValueError('Missing actual PPPM solver settings')
        median=statistics.median(v['ms_per_step'] for v in values)
        points.append(dict(coulomb_cutoff=cutoff,median_ms=median,speedup=median/rbmd,
                           repeats=len(values),solver_settings=[v['solver_settings'] for v in values]))
    best=min(points,key=lambda p:p['median_ms'])
    return dict(points=points,best_sampled_pppm=best,rbmd_median_ms=rbmd,rbmd_repeats=len(baseline),
                nodes=rows[0]['nodes'],gpus=rows[0]['gpus'],n_atoms=audits[0]['n_atoms'],
                data_sha256=audits[0]['data_sha256'],
                scope='Best of sampled Coulomb cutoffs at PPPM requested accuracy 1e-4; fixed LJ cutoff; no equal-RMS claim')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.root),indent=2,allow_nan=False)+'\n')
    print(a.output)

if __name__=='__main__':main()
