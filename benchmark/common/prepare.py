#!/usr/bin/env python3
"""Prepare review experiments; never submits a job or overwrites a run."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from read_kspace import settings
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {'force': '01_force_accuracy', 'fullbox': '02_full_box',
               'dynamics': '04_dynamics_rerun',
               'timing': '05_timing_scaling', 'ablation': '06_ablation', 'tuning': '07_pppm_tuning'}

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def load_system(path, allow_replicate=False):
    s = json.loads(path.read_text())
    if s['name'] not in ('lj', 'water'):
        raise ValueError('NewBenchmark supports lj and water only')
    c = s['rbmd']
    if c['execution']['ensemble'] != 'NVT':
        raise ValueError('Only NVT is in scope; NVE energy-drift experiments are excluded')
    h = c['hyper_parameters']
    if not (0 < h['neighbor']['r_core'] < h['neighbor']['cut_off']):
        raise ValueError('Require 0 < r_core < cut_off')
    if 'alpha' in h['coulomb'] or h['coulomb'].get('debug_alpha'):
        raise ValueError('Remove manual alpha/debug_alpha; scan cutoff with automatic alpha')
    if h['coulomb'].get('rbe_low_modes', False):
        raise ValueError('These experiments test full-box Eq.(9); low-mode variants need a separate label')
    if not allow_replicate and c['init_configuration']['read_data'].get('replicate', [1,1,1]) != [1,1,1]:
        raise ValueError('Use an explicit materialized data file shared by both codes')
    return s

def lammps_setup(s, cfg, accuracy, coulomb_cutoff=None):
    h = cfg['hyper_parameters']; d = cfg['init_configuration']['read_data']
    rs = h['neighbor']['cut_off']; skin = h['neighbor'].get('skin', 1.0)
    ff = h['force_field']; ex = h.get('extend', {})
    lines = [f"units {d['unit'].lower()}", f"atom_style {d['atom_style']}", 'boundary p p p']
    lines += s.get('bonded_styles', [])
    special = ex.get('special_bonds', [0,0,0])
    lines += ['special_bonds lj/coul ' + ' '.join(map(str, special)),
              f'pair_style lj/cut/coul/long {rs} {rs if coulomb_cutoff is None else coulomb_cutoff}',
              'read_data ' + '"' + d['file'] + '"',
              'pair_modify mix ' + ff.get('mixing_rule', 'arithmetic') + ' shift no tail no table 0',
              'dielectric 1', f'kspace_style pppm {accuracy}', 'kspace_modify order 5',
              f'neighbor {skin} bin', 'neigh_modify every 1 delay 0 check yes',
              f"timestep {cfg['execution']['timestep']}", 'reset_timestep 0',
              'thermo_modify norm no lost error flush yes format float %.17g']
    replicas=d.get('replicate',[1,1,1])
    if replicas != [1,1,1]:
        index=next(i for i,line in enumerate(lines) if line.startswith('read_data '))+1
        lines.insert(index,'replicate '+' '.join(map(str,replicas))+(' bond/periodic' if s.get('bonded_styles') else ''))
    return '\n'.join(lines) + '\n'

def thermostat(s, cfg):
    e = cfg['execution']
    t0,t1,tdamp = e['temperature']
    # NoseHooverController uses _mtchain=3 and _nc_tchain=1.
    return (f'fix thermostat all nvt temp {t0} {t1} {tdamp} tchain 3 tloop 1\n'
            + '\n'.join(s.get('constraints', [])) + '\n')

def add_case(root, name, s, cfg, engine, text=None, warmup=0, measured=0, sync='off', coulomb_cutoff=None, lammps_role='production', purpose=None):
    path = root / name; path.mkdir(parents=True)
    write_json(path/'rbmd.json', cfg)
    if text is not None:
        (path/'run.in').write_text(text)
    meta = dict(system=s['name'], engine=engine, warmup=warmup, measured_steps=measured,
                launch_blocking=sync, physical_parameters=cfg['hyper_parameters'],
                timestep=cfg['execution']['timestep'], units=cfg['init_configuration']['read_data']['unit'],
                data=cfg['init_configuration']['read_data']['file'], status='PREPARED_NOT_RUN',
                alpha_scan=bool(cfg['hyper_parameters']['coulomb'].get('debug_alpha')),
                lammps_coulomb_cutoff=coulomb_cutoff,lammps_role=lammps_role,purpose=purpose,
                replicate=cfg['init_configuration']['read_data'].get('replicate',[1,1,1]),
                equilibration_status=s.get('equilibration_status','UNVERIFIED'))
    write_json(path/'case.json', meta)
    cmd = shlex.join(['bash', str(ROOT/'common/run_case.sh'), engine, str(path.resolve())])
    (path/'run.sh').write_text('#!/bin/bash\nset -euo pipefail\n' + cmd + '\n')
    return path

def quiet(cfg, steps, isolated=False):
    cfg['execution']['num_steps'] = steps
    cfg['execution']['diagnostics'] = dict(isolated_timing=isolated, forces=False, energy=False,
        thermostat_heat=False, interval=max(steps,1), prefix='revision')
    cfg['outputs'] = {'thermo_out': {'interval': max(steps+1,1)}}
    cfg['analysis_postprocess'] = {'enabled': False}
    for k in ('neighbor','coulomb'):
        cfg['hyper_parameters'][k]['energy_rbl_flag' if k=='neighbor' else 'energy_rbe_flag'] = 'no'

def prepare(a):
    if a.experiment not in EXPERIMENTS:
        raise ValueError('Experiment excluded from this benchmark: ' + a.experiment)
    s = load_system(a.system, allow_replicate=a.experiment=='timing'); base = s['rbmd']
    if a.cutoffs is None: a.cutoffs=[base['hyper_parameters']['neighbor']['cut_off']] if s['name']=='water' else [8,12,16]
    if s['name']=='water' and a.experiment in ('force','fullbox') and any(rs <= base['hyper_parameters']['neighbor']['r_core'] for rs in a.cutoffs):
        raise ValueError('Water cutoff must exceed configured core radius')
    if a.replicate:
        if a.experiment!='timing' or min(a.replicate)<1: raise ValueError('Positive replication is allowed only for performance timing')
        base['init_configuration']['read_data']['replicate']=a.replicate
    replicas=base['init_configuration']['read_data'].get('replicate',[1,1,1])
    if len(replicas)!=3 or any(type(v)!=int or v<1 for v in replicas): raise ValueError('Invalid replicate factors')
    if a.purpose != 'rdf' or a.damping is not None:
        raise ValueError('This package supports RDF trajectories only')
    if a.experiment == 'dynamics' and a.purpose=='rdf' and s['name']=='water':
        if (base['init_configuration']['read_data']['unit'].upper() != 'REAL' or
                base['execution']['temperature'][2] != 5):
            raise ValueError('Water accuracy trajectories retain 5 fs damping; use systems/*.rdf.json')
    if a.data:
        base['init_configuration']['read_data']['file'] = str(a.data.resolve())
    data = Path(base['init_configuration']['read_data']['file'])
    if not a.allow_missing_data and not data.is_file():
        raise ValueError(f'Missing data file: {data}')
    if a.experiment in ('force','fullbox','ablation') and a.kmax is None:
        raise ValueError('--kmax Kx Ky Kz required; converge deterministic EWALD separately')
    if a.states and (a.experiment!='dynamics' or len(a.states)!=a.repeats):
        raise ValueError('--states is for dynamics; provide one independently equilibrated data file per repeat')
    if a.states and not all(p.is_file() for p in a.states):
        raise ValueError('Missing independently equilibrated state')
    if a.steps <= 0 or a.samples <= 1 or a.repeats <= 0 or a.warmup < 0:
        raise ValueError('Positive steps/repeats, samples > 1, warmup >= 0 required')
    if any(v <= 0 for v in a.P+a.p+(a.kmax or [1])+a.cutoffs):
        raise ValueError('All grid values must be positive')
    alphas=[None]
    if a.alpha_log:
        if a.experiment!='force' or len(a.cutoffs)!=1:
            raise ValueError('--alpha-log requires force mode and exactly one fixed cutoff')
        auto=settings(a.alpha_log.read_text())
        if auto.get('alpha_source')!='auto' or 'alpha' not in auto:
            raise ValueError('--alpha-log must contain an actual automatic alpha')
        if any(not math.isfinite(v) or v<=0 for v in a.alpha_factors):
            raise ValueError('Alpha factors must be finite and positive')
        alphas=[auto['alpha']*v for v in a.alpha_factors]
        if len(set(alphas))!=len(alphas):raise ValueError('Duplicate alpha factors')
    if a.output.exists():
        raise ValueError('Output already exists; preserve previous evidence with a new output path')
    a.output.mkdir(parents=True)
    write_json(a.output/'system.json', s)
    cases = []
    if a.experiment in ('force','fullbox'):
        for rs in a.cutoffs:
            cfg = copy.deepcopy(base); h = cfg['hyper_parameters']
            h['neighbor'].update(cut_off=rs, r_core=base['hyper_parameters']['neighbor']['r_core'] if s['name']=='water' else rs/2)
            h['coulomb']['kmax'] = a.kmax
            quiet(cfg, 0)
            for accuracy in (1e-4, 1e-6, 1e-7):
                text = lammps_setup(s,cfg,accuracy) + 'thermo_style custom step atoms pe\nthermo_modify norm no lost error format float %.17g\nthermo 1\nrun 0\nwrite_dump all custom reference.dump id x y z fx fy fz modify sort id format float %.17g\n'
                cases.append(add_case(a.output,f'rs{rs:g}/pppm_{accuracy:g}',s,cfg,'lammps',text,lammps_role='reference'))
            combos = [('vl_ewald','Verlet-List','EWALD',a.P[0],a.p[0])]
            combos += [('rbe','Verlet-List','RBE',P,a.p[0]) for P in a.P]
            combos += [('rbl','RBL','EWALD',a.P[0],p) for p in a.p]
            combos += [('combined','RBL','RBE',P,p) for P,p in itertools.product(a.P,a.p)]
            for (label,nb,lr,P,p),alpha in itertools.product(combos,alphas):
                c = copy.deepcopy(cfg)
                if alpha is not None:c['hyper_parameters']['coulomb'].update(debug_alpha=True,alpha=alpha)
                c['hyper_parameters']['neighbor'].update(type=nb,neighbor_sample_num=p)
                c['hyper_parameters']['coulomb'].update(type=lr,coulomb_sample_num=P,random_seed=a.seed)
                c['execution']['diagnostics'].update(fixed_samples=a.samples if label!='vl_ewald' else 2,forces=True,interval=1)
                cases.append(add_case(a.output,f'rs{rs:g}/{label}_P{P}_p{p}'+(f'_alpha{alpha:.12g}' if alpha is not None else ''),s,c,'rbmd'))
    elif a.experiment == 'dynamics':
        for rep in range(a.repeats):
            cfg = copy.deepcopy(base); quiet(cfg,a.steps)
            if a.states: cfg['init_configuration']['read_data']['file']=str(a.states[rep].resolve())
            cfg['hyper_parameters']['coulomb']['random_seed']=a.seed+rep
            cfg['hyper_parameters']['neighbor']['rbl_random_seed']=a.seed+rep
            cfg['execution']['diagnostics'].update(energy=False,thermostat_heat=False,interval=a.stride)
            cfg['outputs']={'thermo_out':{'interval':a.stride},'trajectory_out':{'interval':a.stride,'mpi_sharded':True}}
            text = lammps_setup(s,cfg,1e-4)+thermostat(s,cfg)
            text += (f'thermo_style custom step time atoms temp pe ke etotal\nthermo_modify norm no lost error format float %.17g\nthermo {a.stride}\n'
                f'dump trajectory all custom {a.stride} trajectory.dump id type q x y z xu yu zu vx vy vz\n'
                'dump_modify trajectory sort id format float %.17g\n'
                f'run {a.steps}\n')
            cases.append(add_case(a.output,f'rep{rep+1}/rbmd',s,cfg,'rbmd',purpose=a.purpose))
            cases.append(add_case(a.output,f'rep{rep+1}/lammps',s,cfg,'lammps',text,purpose=a.purpose))
    else:
        variants=[('combined','RBL','RBE')]
        if a.experiment=='ablation':
            variants=[('vl_ewald','Verlet-List','EWALD'),('rbe','Verlet-List','RBE'),('rbl','RBL','EWALD'),('combined','RBL','RBE')]
        cutoffs=a.cutoffs if a.experiment=='tuning' else [base['hyper_parameters']['neighbor']['cut_off']]
        for rs,rep in itertools.product(cutoffs,range(a.repeats)):
            cfg=copy.deepcopy(base); quiet(cfg,a.warmup+a.steps,a.isolated)
            # PPPM tuning changes only its Coulomb cutoff; preserve the LJ Hamiltonian and RBMD baseline.
            cfg['hyper_parameters']['coulomb']['random_seed']=a.seed+rep
            for label,nb,lr in variants:
                if a.experiment=='tuning' and rs!=cutoffs[0]:continue
                c=copy.deepcopy(cfg); c['hyper_parameters']['neighbor']['type']=nb
                c['hyper_parameters']['coulomb']['type']=lr
                if a.kmax: c['hyper_parameters']['coulomb']['kmax']=a.kmax
                cases.append(add_case(a.output,(f'rbmd/rep{rep+1}/{label}' if a.experiment=='tuning' else f'rs{rs:g}/rep{rep+1}/{label}'),s,c,'rbmd',warmup=a.warmup,measured=a.steps,sync='on' if a.isolated else 'off'))
            coulomb_cutoff=rs if a.experiment=='tuning' else None
            text=lammps_setup(s,cfg,1e-4,coulomb_cutoff)+thermostat(s,cfg)
            text += (f'thermo_style custom step atoms temp\nthermo_modify norm no lost error format float %.17g\nthermo {a.warmup+a.steps+1}\ntimer full nosync\n'
                     f'run {a.warmup}\nrun {a.steps} pre no post yes\n')
            cases.append(add_case(a.output,f'rs{rs:g}/rep{rep+1}/lammps',s,cfg,'lammps',text,a.warmup,a.steps,'on' if a.isolated else 'off',coulomb_cutoff=coulomb_cutoff))
    (a.output/'cases.txt').write_text(''.join(str(p.resolve())+'\n' for p in cases))
    write_json(a.output/'manifest.json',dict(experiment=a.experiment,system=s['name'],cases=len(cases),
        source_system_sha256=hashlib.sha256(a.system.read_bytes()).hexdigest(),
        fixed_snapshot= str(data),prepared_only=True,seed=a.seed,
        independent_state_files_supplied=bool(a.states),independence_verified=False,reference_is_converged=False,
        purpose=a.purpose if a.experiment=='dynamics' else None,replicate=replicas,
        alpha_scan=bool(a.alpha_log),alpha_values=alphas,
        alpha_log_sha256=hashlib.sha256(a.alpha_log.read_bytes()).hexdigest() if a.alpha_log else None))
    print(f'Prepared {len(cases)} cases in {a.output}; no jobs submitted')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=EXPERIMENTS)
    p.add_argument('--system',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--replicate',type=int,nargs=3,help='Timing only: both engines replicate the same base data')
    p.add_argument('--purpose',choices=['rdf'],default='rdf')
    p.set_defaults(damping=None)
    p.add_argument('--states',type=Path,nargs='+')
    p.add_argument('--allow-missing-data',action='store_true',help='Offline generation only; execution still checks data')
    p.add_argument('--P',type=int,nargs='+',default=[100,200,400,800])
    p.add_argument('--p',type=int,nargs='+',default=[30,60,120])
    p.add_argument('--cutoffs',type=float,nargs='+',help='Default: water 9; LJ 8 12 16. Tuning scans only the PPPM Coulomb cutoff.')
    p.add_argument('--kmax',type=int,nargs=3)
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--steps',type=int,help='Measured steps, excluding warmup: default 1000')
    p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--repeats',type=int,help='Default: 5 for dynamics, 1 for performance experiments')
    p.add_argument('--stride',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--isolated',action='store_true')
    p.add_argument('--alpha-log',type=Path,help='Force-only alpha sweep around automatic alpha read from this stdout log')
    p.add_argument('--alpha-factors',type=float,nargs='+',default=[0.5,0.75,1,1.5,2])
    a=p.parse_args()
    if a.repeats is None:
        a.repeats=3 if a.experiment=='dynamics' else 1
    if a.steps is None:
        a.steps=1000
    if a.stride<=0: p.error('stride must be positive')
    try: prepare(a)
    except (ValueError,KeyError) as e: p.error(str(e))

if __name__=='__main__': main()
