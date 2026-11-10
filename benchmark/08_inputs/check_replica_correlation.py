#!/usr/bin/env python3
"""Measure positional replica correlation in a full-box dump; not an equilibration certificate."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from merge_trajectory_shards import read_frame

def measure(frame,factors):
    step,box,_,fields,atoms=frame
    if box[0].split()[3:]!=['pp','pp','pp']:raise ValueError('Periodic orthogonal box required')
    bounds=[tuple(map(float,row.split())) for row in box[1:]]
    if any(len(b)!=2 or not all(math.isfinite(v) for v in b) or b[1]<=b[0] for b in bounds):raise ValueError('Invalid box')
    if len(factors)!=3 or any(type(v)!=int or v<1 for v in factors) or math.prod(factors)<=1: raise ValueError('Invalid replication')
    counts=math.prod(factors)
    if len(atoms)%counts:raise ValueError('Atom count not divisible by replication')
    base=len(atoms)//counts;positions={}
    for line in atoms:
        row=dict(zip(fields,line.split()));atom=int(row['id'])
        if atom in positions:raise ValueError('Duplicate atom ID')
        positions[atom]=[float(row[a]) for a in ('x','y','z')]
    if any(not math.isfinite(v) for point in positions.values() for v in point): raise ValueError('Nonfinite coordinates')
    if set(positions)!=set(range(1,len(atoms)+1)):raise ValueError('Requires contiguous LAMMPS replicate IDs; remapped IDs need an explicit map')
    lengths=[(hi-lo)/f for (lo,hi),f in zip(bounds,factors)]
    error=0.;maximum=0.
    for atom in range(base+1,len(atoms)+1):
        origin=positions[(atom-1)%base+1];point=positions[atom]
        d2=sum(((v-w)-L*round((v-w)/L))**2 for v,w,L in zip(point,origin,lengths))
        error+=d2;maximum=max(maximum,math.sqrt(d2))
    return dict(step=step,base_atoms=base,copies=counts,folded_rms=math.sqrt(error/(len(atoms)-base)),folded_max=maximum,
        caveat='Near-zero implies identical positional copies under the stated ID mapping; nonzero alone does not prove decorrelation/equilibration')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dump',type=Path)
    p.add_argument('--replicate',type=int,nargs=3,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.replicate)<1 or math.prod(a.replicate)<=1:p.error('Replication product must exceed one')
    rows=[]
    with a.dump.open() as f:
        while (frame:=read_frame(f)) is not None:rows.append(measure(frame,a.replicate))
    if not rows:raise ValueError('No frames')
    a.output.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
