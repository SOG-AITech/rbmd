#!/usr/bin/env python3
"""Validate built-in online moments and compare sorted PPPM dumps in bounded memory."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
from read_kspace import settings
spec=importlib.util.spec_from_file_location('force_analysis',ROOT/'01_force_accuracy/analyze.py')
force_analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(force_analysis)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def dump_rows(path):
    with path.open() as f:
        def line():
            value=f.readline()
            if not value:raise ValueError(f'Truncated reference: {path}')
            return value.strip()
        if line()!='ITEM: TIMESTEP' or int(line())!=0:raise ValueError('Expected reference timestep 0')
        if line()!='ITEM: NUMBER OF ATOMS':raise ValueError('Missing atom count')
        n=int(line())
        if n<=0 or line()!='ITEM: BOX BOUNDS pp pp pp':raise ValueError('Expected nonempty periodic orthogonal box')
        bounds=[tuple(map(float,line().split())) for _ in range(3)]
        if any(len(b)!=2 or not all(map(math.isfinite,b)) or b[1]<=b[0] for b in bounds):raise ValueError('Invalid bounds')
        header=line().split()
        if header[:2]!=['ITEM:','ATOMS']:raise ValueError('Missing atom columns')
        fields=header[2:]
        keys=['id','x','y','z','fx','fy','fz']
        if len(fields)!=len(set(fields)) or not set(keys)<=set(fields):raise ValueError('Invalid columns')
        indices=[fields.index(k) for k in keys]
        yield (n,bounds)
        previous=None
        for _ in range(n):
            words=line().split()
            if len(words)!=len(fields):raise ValueError('Malformed reference row')
            atom=int(words[indices[0]]); values=tuple(float(words[i]) for i in indices[1:])
            if previous is not None and atom<=previous:raise ValueError('Reference IDs must be unique and sorted')
            if not all(map(math.isfinite,values)):raise ValueError('Nonfinite reference')
            previous=atom
            yield (atom,values)
        if f.read(1):raise ValueError('Expected exactly one reference frame')

def compare_references(group):
    names=['pppm_0.0001','pppm_1e-06','pppm_1e-07']
    data_hashes=set();solver_settings={}
    for name in names:
        case=group/name;force_analysis.verify_reference(case)
        receipt=json.loads((case/'run_receipt.json').read_text())
        if receipt['input_sha256']!=digest(case/'run.in'):raise ValueError('Reference input changed after run')
        data_hashes.add(json.loads((case/'preflight.json').read_text())['data_sha256'])
        solver_settings[name]=settings((case/'stdout.log').read_text())
    if len(data_hashes)!=1:raise ValueError('Reference data hashes differ')
    streams=[dump_rows(group/name/'reference.dump') for name in names]
    headers=[next(s) for s in streams]
    if not headers[0]==headers[1]==headers[2]:raise ValueError('Reference boxes/counts differ')
    n=headers[0][0];norm=0.;errors=[0.,0.];seen=0
    for rows in itertools.zip_longest(*streams):
        if any(row is None for row in rows) or len({row[0] for row in rows})!=1:raise ValueError('Reference ID sets differ')
        ref=rows[2][1]
        for i in range(2):
            if any(abs(a-b)>1e-10 for a,b in zip(rows[i][1][:3],ref[:3])):raise ValueError('Reference positions differ')
            errors[i]+=sum((a-b)**2 for a,b in zip(rows[i][1][3:],ref[3:]))
        norm+=sum(v*v for v in ref[3:]);seen+=1
    if seen!=n or not math.isfinite(norm) or norm<=0 or not all(map(math.isfinite,errors)):raise ValueError('Invalid reference statistics')
    return dict(n_atoms=n,reference_data_sha256=data_hashes.pop(),reference_solver_settings=solver_settings,reference_mean_squared_force=norm/n,
        measured_pppm_1e4_relative_rms=math.sqrt(errors[0]/norm),
        reference_1e6_vs_1e7_relative_rms=math.sqrt(errors[1]/norm),
        reference_absolute_rms=math.sqrt(errors[1]/n),
        reference_sha256=digest(group/'pppm_1e-07/reference.dump'))

def validate_moments(s,expected):
    if s.get('schema')!=1 or s['n_samples']!=expected or s['n_atoms']<=0:raise ValueError('Incomplete online statistics')
    keys=('mse','rms_force_error','relative_rms_force_error','rms_bias','variance_population','variance_unbiased')
    if any(type(s.get(k)) not in (int,float) or not math.isfinite(s[k]) or s[k]<0 for k in keys):raise ValueError('Invalid online statistics')
    def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-20)
    if not close(s['mse'],s['rms_force_error']**2):raise ValueError('RMS/MSE mismatch')
    if not close(s['mse'],s['rms_bias']**2+s['variance_population']):raise ValueError('Bias/variance decomposition mismatch')
    if not close(s['variance_unbiased'],s['variance_population']*expected/(expected-1)):raise ValueError('Unbiased variance mismatch')

def analyze_group(group,fraction=.1):
    reference=compare_references(group);records=[]
    for method in ('rbe','combined'):
        case=group/method
        cfg=json.loads((case/'rbmd.json').read_text());d=cfg['execution']['diagnostics']
        receipt=json.loads((case/'run_receipt.json').read_text())
        pre=json.loads((case/'preflight.json').read_text())
        if receipt['exit_code']!=0 or pre['status']!='PASS' or receipt['input_sha256']!=digest(case/'rbmd.json'):raise ValueError('Run/configuration validation failed')
        if d['forces'] or list(case.glob('revision.forces.rank*.csv')):raise ValueError('03 must not output per-atom CSV')
        s=json.loads((case/'revision.force_statistics.json').read_text())
        validate_moments(s,d['fixed_samples'])
        if pre['data_sha256']!=reference['reference_data_sha256']:raise ValueError('RBMD/reference input hashes differ')
        if s['n_atoms']!=reference['n_atoms'] or s['n_atoms']!=pre['n_atoms']:raise ValueError('Atom count mismatch')
        if Path(s['reference_force_file']).resolve()!=(group/'pppm_1e-07/reference.dump').resolve():raise ValueError('Wrong reference file')
        if (s['reference_timestep']!=0 or s['reference_id_offset']!=-1 or
            s['reference_position_tolerance']!=d['reference_position_tolerance']):raise ValueError('Wrong reference matching options')
        if not math.isclose(s['reference_mean_squared_force'],reference['reference_mean_squared_force'],rel_tol=1e-9):raise ValueError('Reference force norm mismatch')
        if not math.isclose(s['relative_rms_force_error'],s['rms_force_error']/math.sqrt(reference['reference_mean_squared_force']),rel_tol=1e-9):raise ValueError('Relative RMS mismatch')
        with (case/'revision.force_errors.csv').open() as f:
            rows=csv.DictReader(f);last=None;count=0
            for count,row in enumerate(rows,1):
                if int(row['sample'])!=count or int(row['n_atoms'])!=s['n_atoms']:raise ValueError('Online sample sequence mismatch')
                last=row
        if count!=s['n_samples'] or not math.isclose(float(last['cumulative_rms_force_error']),s['rms_force_error'],rel_tol=1e-12):raise ValueError('Incomplete sample CSV')
        actual=settings((case/'stdout.log').read_text());c=cfg['hyper_parameters']['coulomb']
        if actual.get('alpha_source')!='manual' or not math.isclose(actual['alpha'],c['alpha'],rel_tol=1e-5):raise ValueError('Actual alpha mismatch')
        threshold=fraction*s['rms_force_error'];converged=reference['reference_absolute_rms']<=threshold
        s.update(reference,parameters=cfg['hyper_parameters'],solver_settings=actual,
            reference_converged=converged,reference_error_fraction=fraction,
            reference_allowed_absolute_rms=threshold,accuracy_status='REFERENCE_ADEQUATE' if converged else 'REFERENCE_LIMITED',
            binary_sha256=receipt['binary_sha256'],statistics_source='RBMD_ONLINE_WELFORD',case=str(case))
        (case/'force_statistics.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n');records.append(s)
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--reference-error-fraction',type=float,default=.1);a=p.parse_args()
    if not 0<a.reference_error_fraction<1:p.error('Invalid reference error fraction')
    groups=sorted(a.root.glob('N*/rs*')) or sorted(a.root.glob('rs*'))
    if not groups:raise ValueError('No force groups')
    records=[r for g in groups for r in analyze_group(g,a.reference_error_fraction)]
    (a.root/'online-summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    if not all(r['reference_converged'] for r in records):raise SystemExit('Reference not converged; do not claim 03 validation')
    print(f'{len(records)} online force cases validated')
if __name__=='__main__':main()
