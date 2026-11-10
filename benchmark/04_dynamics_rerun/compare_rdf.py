#!/usr/bin/env python3
"""Average LAMMPS fix ave/time RDF frames; compare identical radial grids."""
import argparse
import csv
import json
import math
from pathlib import Path

def load(path):
    lines=iter(line for line in path.read_text().splitlines() if line.strip() and not line.startswith('#'))
    frames=[];radii=None;steps=[]
    for header in lines:
        step,n=map(int,header.split());steps.append(step);grid=[];data=[]
        for i in range(n):
            row=list(map(float,next(lines).split()))
            if int(row[0])!=i+1 or len(row)<4 or not all(math.isfinite(v) for v in row):raise ValueError('Malformed RDF row')
            grid.append(row[1]);data.append(row[2::2])
        if radii is None:radii=grid
        if grid!=radii:raise ValueError('RDF grid changed')
        frames.append(data)
    if not frames:raise ValueError('No RDF frames')
    avg=[[sum(f[i][j] for f in frames)/len(frames) for j in range(len(frames[0][i]))] for i in range(len(radii))]
    return radii,avg,steps

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rbmd',type=Path);p.add_argument('pppm',type=Path)
    p.add_argument('--peak-min',type=float,required=True);p.add_argument('--peak-max',type=float,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    x,y,sy=load(a.rbmd);xx,z,sz=load(a.pppm)
    if x!=xx or len(y[0])!=len(z[0]) or sy!=sz:raise ValueError('Use the same bins, pair ordering and sampled timesteps in both methods')
    keep=[i for i,r in enumerate(x) if a.peak_min<=r<=a.peak_max]
    if not keep:raise ValueError('Empty peak window')
    result=[]
    for j in range(len(y[0])):
        ir=max(keep,key=lambda i:y[i][j]);ip=max(keep,key=lambda i:z[i][j]);ref=z[ip][j]
        delta=(y[ir][j]/ref-1) if ref else None
        result.append(dict(pair_index=j,rbmd_peak_r=x[ir],pppm_peak_r=x[ip],peak_shift=x[ir]-x[ip],
            peak_relative_deviation=delta,within_5_percent=delta is not None and abs(delta)<=.05,
            curve_rms=math.sqrt(sum((b[j]-c[j])**2 for b,c in zip(y,z))/len(x))))
    a.output.write_text(json.dumps({'pairs':result,'frames':len(sy),'first':sy[0],'last':sy[-1],'warning':'Per-trajectory comparison; use five independent trajectories for uncertainty. Peak height alone does not certify force accuracy.'},indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
