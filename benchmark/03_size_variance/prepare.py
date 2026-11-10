#!/usr/bin/env python3
"""Prepare MJ2(a) fixed-configuration size scans; never run or submit jobs."""
import argparse,copy,json,math,os,re,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'common'),str(Path(__file__).resolve().parents[1]/'08_inputs')]
from prepare import load_system,quiet,add_case,lammps_setup,write_json
from audit_sources import header,digest

DEFAULT_ROOT='/public/home/acymkxdx47/ReviewRBMD/revision_lj_water_20261015/relaxed_systems_20261017'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path(DEFAULT_ROOT))
    p.add_argument('--data',type=Path,nargs='+',required=True,help='Explicit final relaxed data paths, relative to data-root or absolute; one per N')
    p.add_argument('--P',type=int,required=True);p.add_argument('--p',type=int,required=True)
    p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--rs',type=float,required=True);p.add_argument('--rc',type=float,required=True)
    p.add_argument('--samples',type=int,default=1000);p.add_argument('--seed',type=int,default=20261018)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.P<1 or a.p<1 or a.samples<2 or
        not all(math.isfinite(v) and v>0 for v in (a.alpha,a.rs,a.rc)) or a.rc>=a.rs):p.error('Require positive P,p,alpha; samples>=2; 0<rc<rs')
    s=load_system(a.system);root=a.data_root.resolve();inputs=[]
    for path in a.data:
        path=Path(os.path.abspath(root/path if not path.is_absolute() else path))
        if root not in path.parents or not path.is_file():p.error(f'Input must be an existing entry inside data-root: {path}')
        source_manifest=path.parent/'manifest.json'
        source=json.loads(source_manifest.read_text())
        if any(v in source.get('status','').upper() for v in ('PREPARING','NOT_FINAL')):p.error(f'Input preparation is not final: {path}')
        family='L' if s['name']=='lj' else 'W' if s['name']=='water' else None
        if family is None or not source['system'].startswith(family):p.error('Do not mix physical systems in a size scan')
        n=header(path)
        if not n or n!=source['N']:p.error(f'Atom count differs from source manifest: {path}')
        with path.open() as f:head=f.read(65536)
        required=['Pair Coeffs']+(['Bond Coeffs','Angle Coeffs'] if family=='W' else [])
        if any(not re.search(r'^'+name+r'\b',head,re.M) for name in required):p.error('Missing coefficients; select the registered test_input.data representation')
        if re.search(r'\bxy xz yz\b',head):p.error('Only orthogonal boxes are supported')
        bounds=[]
        for axis in 'xyz':
            match=re.search(r'^\s*(\S+)\s+(\S+)\s+'+axis+'lo '+axis+r'hi\s*$',head,re.M)
            if not match:p.error('Missing box bounds')
            lo,hi=map(float,match.groups())
            if not math.isfinite(hi-lo) or hi<=lo:p.error('Invalid box')
            bounds.append([lo,hi])
        rho=n/math.prod(hi-lo for lo,hi in bounds)
        if inputs and not math.isclose(rho,inputs[0]['number_density'],rel_tol=1e-5):p.error('Size scan requires common number density')
        if min(hi-lo for lo,hi in bounds)<=2*a.rs:p.error('Cutoff must be smaller than half every box length')
        inputs.append(dict(N=n,path=str(path.resolve()),entry=str(path),sha256=digest(path),
            number_density=rho,bounds=bounds,source_manifest=source,source_manifest_sha256=digest(source_manifest)))
    if len(inputs)<2 or len({r['N'] for r in inputs})!=len(inputs):p.error('Select >=2 distinct N, exactly one final configuration per N')
    if a.output.exists():p.error('Output exists; use a new directory')
    a.output.mkdir(parents=True);cases=[]
    for entry in sorted(inputs,key=lambda r:r['N']):
        group=a.output/f"N{entry['N']}";cfg=copy.deepcopy(s['rbmd'])
        cfg['init_configuration']['read_data'].update(file=entry['path'],replicate=[1,1,1])
        h=cfg['hyper_parameters'];h['neighbor'].update(cut_off=a.rs,r_core=a.rc,neighbor_sample_num=a.p,rbl_random_seed=a.seed)
        h['coulomb'].update(type='RBE',coulomb_sample_num=a.P,random_seed=a.seed)
        h['coulomb'].pop('kmax',None)
        quiet(cfg,0)
        for accuracy in (1e-4,1e-6,1e-7):
            text=lammps_setup(s,cfg,accuracy).replace('kspace_modify order 5','kspace_modify order 7')+'thermo_style custom step atoms pe\nthermo_modify norm no lost error format float %.17g\nrun 0\nwrite_dump all custom reference.dump id x y z fx fy fz modify sort id format float %.17g\n'
            cases.append(add_case(group,f'rs{a.rs:g}/pppm_{accuracy:g}',s,cfg,'lammps',text,lammps_role='reference',purpose='MJ2(a)'))
        for label,neighbor,coulomb in [('rbe','Verlet-List','RBE'),('combined','RBL','RBE')]:
            c=copy.deepcopy(cfg);c['hyper_parameters']['neighbor']['type']=neighbor
            c['hyper_parameters']['coulomb'].update(type=coulomb,debug_alpha=True,alpha=a.alpha)
            # Match the offline FP32 coordinate tolerance, including periodic wrapping.
            scale=max(max(abs(lo),abs(hi),hi-lo) for lo,hi in entry['bounds'])
            c['execution']['diagnostics'].update(fixed_samples=a.samples,forces=False,interval=1,
                reference_force_file=str((group/f'rs{a.rs:g}/pppm_1e-07/reference.dump').resolve()),
                reference_timestep=0,reference_id_offset=-1,
                reference_position_tolerance=1e-6+2**-22*scale)
            cases.append(add_case(group,f'rs{a.rs:g}/{label}',s,c,'rbmd',purpose='MJ2(a)'))
    (a.output/'cases.txt').write_text(''.join(str(c.resolve())+'\n' for c in cases))
    write_json(a.output/'manifest.json',dict(experiment='MJ2(a)',system=s['name'],inputs=inputs,
        fixed_parameters=dict(P=a.P,p=a.p,alpha=a.alpha,rs=a.rs,rc=a.rc,dt=s['rbmd']['execution']['timestep']),
        samples=a.samples,methods=['rbe','combined'],deferred_methods=['rbl'],prepared_only=True,equilibration_verified=False,
        scope='Only selected N; fixed configurations, no trajectory, heat flux or NVE. No maximum-N extrapolation.'))
    print(f'Prepared {len(cases)} cases; no jobs submitted or simulations run')

if __name__=='__main__':main()
