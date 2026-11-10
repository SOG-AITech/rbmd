#!/usr/bin/env python3
"""Make LAMMPS rerun for RBMD or LAMMPS trajectories (fixed orthogonal box)."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from merge_trajectory_shards import read_frame
from prepare import lammps_setup, add_case, load_system, write_json

def atom_identity(data,style):
    if style not in ('charge','full'):raise ValueError('Only charge/full data styles supported')
    atoms={};inside=False
    with data.open() as f:
        for line in f:
            text=line.split('#',1)[0].strip()
            if text=='Atoms':inside=True;continue
            if not inside or not text:continue
            fields=text.split()
            if not fields[0].isdigit():break
            offset=1 if style=='full' else 0
            atom=int(fields[0])
            if atom in atoms:raise ValueError('Duplicate data atom ID')
            atoms[atom]=(int(fields[1+offset]),float(fields[2+offset]),int(fields[1]) if offset else None)
    if not atoms:raise ValueError('Missing Atoms section')
    return atoms

def convert(source, output, first, stride, tolerance=2e-3, identity=None, water=False):
    """Recover integer images from explicit xu; never infer crossings from adjacent frames."""
    steps=[]; expected=None; initial_box=None
    with source.open() as inp, output.open('x') as out:
        previous=-1
        while True:
            frame=read_frame(inp)
            if frame is None: break
            step,box,header,fields,rows=frame
            if step<=previous: raise ValueError('Trajectory timesteps must strictly increase')
            previous=step
            if step<first or (step-first)%stride: continue
            if 'xy' in box[0] or len(box[0].split())!=6: raise ValueError('Only orthogonal boxes supported; triclinic needs a matrix image conversion')
            bounds=[list(map(float,l.split())) for l in box[1:]]
            if any(len(b)!=2 or not all(math.isfinite(x) for x in b) or b[1]<=b[0] for b in bounds): raise ValueError('Invalid box')
            if initial_box is None: initial_box=bounds
            if bounds!=initial_box: raise ValueError('Fixed-volume postprocessing only; NPT needs a separate displacement/dielectric convention')
            required={'id','type','q','x','y','z','xu','yu','zu','vx','vy','vz'}
            if not required<=set(fields): raise ValueError('Need wrapped AND unwrapped coordinates, charges, types and velocities; do not reconstruct images by minimum displacement')
            mapped=[]; ids=set();molecules={}
            for row in rows:
                d=dict(zip(fields,row.split())); atom=int(d['id'])
                if atom in ids or atom<=0: raise ValueError('Duplicate/nonpositive trajectory ID')
                ids.add(atom); images=[]
                if identity is not None:
                    if atom not in identity:raise ValueError('Trajectory ID absent from data')
                    typ,charge,molecule=identity[atom]
                    if int(d['type'])!=typ or not math.isclose(float(d['q']),charge,rel_tol=1e-6,abs_tol=1e-8):
                        raise ValueError('Trajectory/data atom type or charge mismatch')
                    if water:
                        if not molecule:raise ValueError('Water dielectric requires molecule IDs')
                        molecules.setdefault(molecule,[]).append(d)
                if not all(math.isfinite(float(d[k])) for k in required): raise ValueError('Non-finite trajectory value')
                for k,ku,b in zip(('x','y','z'),('xu','yu','zu'),bounds):
                    L=b[1]-b[0]; delta=float(d[ku])-float(d[k]); im=round(delta/L)
                    if abs(delta-im*L)>tolerance: raise ValueError(f'Unwrapped coordinate inconsistent for atom {atom}; check output precision/images')
                    images.append(im)
                mapped.append(' '.join(d[k] for k in ('id','type','q','x','y','z','vx','vy','vz'))+' '+' '.join(map(str,images))+'\n')
            if identity is not None and ids!=set(identity):raise ValueError('Trajectory/data ID sets differ')
            for molecule,parts in molecules.items():
                if len(parts)!=3 or abs(sum(float(d['q']) for d in parts))>1e-6:
                    raise ValueError('This water dielectric setup requires neutral three-site molecules')
                for d in parts[1:]:
                    for k,ku,b in zip(('x','y','z'),('xu','yu','zu'),bounds):
                        L=b[1]-b[0];delta=float(d[k])-float(parts[0][k]);delta-=round(delta/L)*L
                        if abs(float(d[ku])-float(parts[0][ku])-delta)>tolerance:
                            raise ValueError(f'Water molecule {molecule} has inconsistent unwrapped images')
            if expected is None: expected=ids
            if ids!=expected: raise ValueError('Atom ID set changed between frames; full trajectory required')
            out.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n');out.writelines(box)
            out.write('ITEM: ATOMS id type q x y z vx vy vz ix iy iz\n');out.writelines(mapped);steps.append(step)
    if len(steps)<2 or any(b-a!=stride for a,b in zip(steps,steps[1:])):
        raise ValueError('Need at least two frames at the exact requested stride (no gaps)')
    if steps[0]!=first: raise ValueError('Requested first frame is absent')
    return steps, len(expected)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',type=Path,required=True);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first',type=int,required=True);p.add_argument('--stride',type=int,required=True)
    p.add_argument('--observable',choices=['rdf','dielectric','structure'],default='rdf')
    p.add_argument('--radius',type=float);p.add_argument('--dr',type=float,default=.1)
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists; choose a new directory')
    if a.stride<=0 or a.dr<=0: p.error('stride and dr must be positive')
    if a.first<0 or a.first%a.stride: p.error('first must be nonnegative and a multiple of stride for fix ave/time')
    s=load_system(a.system);cfg=json.loads((a.case/'rbmd.json').read_text())
    if json.loads((a.case/'case.json').read_text())['system']!=s['name']: p.error('Case and system mismatch')
    supported={'dielectric':'water','structure':'lj'}
    if a.observable in supported and s['name']!=supported[a.observable]:
        p.error(f'{a.observable} is assigned to {supported[a.observable]} in this suite')
    target_temperature=None
    if a.observable=='dielectric':
        temperatures=cfg['execution'].get('temperature',[])
        if (len(temperatures)<2 or any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in temperatures[:2])
            or not math.isclose(temperatures[0],temperatures[1],rel_tol=1e-8)):
            p.error('Dielectric analysis requires a constant positive thermostat target temperature')
        target_temperature=temperatures[0]
    rs=cfg['hyper_parameters']['neighbor']['cut_off'];bins=None
    if a.observable=='rdf':
        if a.radius is None or not math.isfinite(a.radius) or a.radius<=0:p.error('RDF requires a positive --radius')
        bins=round(a.radius/a.dr)
        if bins<1 or not math.isclose(bins*a.dr,a.radius,abs_tol=1e-8) or a.radius>rs:
            p.error('radius must be an integer multiple of dr and <= force cutoff')
    elif a.radius is not None:p.error('--radius only applies to RDF')
    # Separate rerun folder: the original trajectory and its physics remain untouched.
    a.output.mkdir(parents=True)
    trajectory=a.output/'trajectory.images.dump'
    data=cfg['init_configuration']['read_data']
    identity=atom_identity(Path(data['file']),data['atom_style'])
    steps,n=convert(a.trajectory,trajectory,a.first,a.stride,identity=identity,water=a.observable=='dielectric')
    text=lammps_setup(s,cfg,1e-6)
    text+=f'read_dump "{trajectory.resolve()}" {a.first} x y z vx vy vz ix iy iz box yes replace yes\n'
    fields='step time atoms'
    if a.observable=='rdf':
        text+=f'compute rdf_all all rdf {bins} '+' '.join(map(str,s['rdf_pairs']))+f' cutoff {a.radius}\n'
        text+=f'comm_modify cutoff {max(rs,a.radius)+cfg["hyper_parameters"]["neighbor"].get("skin",1)}\n'
        text+=f'fix rdf_output all ave/time {a.stride} 1 {a.stride} c_rdf_all[*] file rdf.dat mode vector\n'
    elif a.observable=='dielectric':
        text+='compute dip all dipole\nvariable charge atom q\ncompute qsum all reduce sum v_charge\n'
        fields+=' c_dip[1] c_dip[2] c_dip[3] c_qsum vol'
    elif a.observable=='structure':
        text+='variable q2 atom q*q\ncompute qnorm all reduce sum v_q2\n'
        for mode in (1,2,3,4):
            for axis in ('x','y','z'):
                tag=f'{axis}{mode}'
                text+=f'variable re{tag} atom q*cos(2*PI*{mode}*{axis}/l{axis})\nvariable im{tag} atom q*sin(2*PI*{mode}*{axis}/l{axis})\n'
                text+=f'compute rho{tag} all reduce sum v_re{tag} v_im{tag}\n'
                text+=f'variable S{tag} equal (c_rho{tag}[1]^2+c_rho{tag}[2]^2)/c_qnorm\n'
                fields+=f' v_S{tag}'
        fields+=' c_qnorm lx ly lz'
    text+=f'thermo_style custom {fields}\nthermo_modify norm no lost error format float %.17g\nthermo {a.stride}\n'
    text+=f'rerun "{trajectory.resolve()}" first {steps[0]} last {steps[-1]} every {a.stride} dump x y z vx vy vz ix iy iz box yes\n'
    path=add_case(a.output,'lammps',s,cfg,'lammps',text)
    write_json(a.output/'postprocess.json',dict(observable=a.observable,system=s['name'],
        source=str(a.trajectory.resolve()),n_atoms=n,first=steps[0],last=steps[-1],frames=len(steps),stride=a.stride,
        timestep=cfg['execution']['timestep'],units=cfg['init_configuration']['read_data']['unit'],
        dr=a.dr if bins else None,radius=a.radius,pppm_requested_accuracy=1e-6,
        target_temperature_K=target_temperature,
        force_scope='physical conservative forces; no thermostat or SHAKE constraint force'))
    print(f'Prepared {a.observable}: {path}; run via common/job.slurm.')

if __name__=='__main__':main()
