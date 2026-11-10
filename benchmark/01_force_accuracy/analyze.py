#!/usr/bin/env python3
"""Compare each fixed sample ensemble to PPPM 1e-7; check 1e-6 convergence."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from analyze_force_samples import analyze, reference_rows
from read_kspace import settings

def compare_dump(sample,reference,tolerance):
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'force.csv'
        with p.open('w') as f:
            w=csv.writer(f); w.writerow(['sample','id','x','y','z','fx','fy','fz'])
            for step,atom,force,position in reference_rows(str(sample)):
                w.writerow([step,atom,*position,*force])
        result,_,_=analyze([str(p)],[str(reference)],position_tolerance=tolerance)
    return result

def verify_reference(case):
    meta=json.loads((case/'case.json').read_text())
    pre=json.loads((case/'preflight.json').read_text())
    receipt=json.loads((case/'run_receipt.json').read_text())
    build=pre['binaries']['lammps']['build']
    if (meta.get('lammps_role')!='reference' or pre.get('status')!='PASS' or
        receipt.get('exit_code')!=0 or build.get('FFT_SINGLE')!='OFF' or build.get('PKG_KOKKOS')!='OFF'):
        raise ValueError(f'Reference must be a completed CPU double run: {case}')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--reference-id-offset',type=int,required=True)
    p.add_argument('--position-tolerance',type=float,default=1e-10,help='Strict fixed-sample movement tolerance')
    p.add_argument('--reference-position-tolerance',type=float,default=1e-6)
    p.add_argument('--reference-position-rtol',type=float,default=2**-22,help='Two FP32 epsilons scaled by reference box/coordinate')
    p.add_argument('--reference-error-fraction',type=float,default=.1)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if any(not math.isfinite(v) or v<0 for v in (a.position_tolerance,a.reference_position_tolerance,a.reference_position_rtol)) or not 0<a.reference_error_fraction<1:p.error('Invalid tolerances or error fraction')
    records=[]
    for group in sorted(a.root.glob('rs*')):
        for name in ('pppm_0.0001','pppm_1e-06','pppm_1e-07'):verify_reference(group/name)
        ref=group/'pppm_1e-07/reference.dump'
        baseline=compare_dump(group/'pppm_0.0001/reference.dump',ref,a.position_tolerance)
        convergence=compare_dump(group/'pppm_1e-06/reference.dump',ref,a.position_tolerance)
        target=baseline['relative_rms_force_error'];delta=convergence['relative_rms_force_error']
        for case in sorted(group.iterdir()):
            if not (case/'rbmd.json').exists() or case.name.startswith('pppm'):continue
            files=sorted(map(str,case.glob('revision.forces.rank*.csv')))
            if not files:raise ValueError(f'Missing samples: {case}')
            r,_,_=analyze(files,[str(ref)],id_offset=a.reference_id_offset,position_tolerance=a.position_tolerance,
                reference_position_tolerance=a.reference_position_tolerance,reference_position_rtol=a.reference_position_rtol)
            ref_error=convergence['rms_force_error']; threshold=a.reference_error_fraction*r['rms_force_error']
            converged=ref_error<=threshold
            cfg=json.loads((case/'rbmd.json').read_text())
            actual=settings((case/'stdout.log').read_text())
            if 'alpha' not in actual:raise ValueError(f'Missing actual alpha: {case}')
            coulomb=cfg['hyper_parameters']['coulomb']
            expected_source='manual' if coulomb.get('debug_alpha') else 'auto'
            if actual.get('alpha_source')!=expected_source:raise ValueError(f'Wrong alpha mode: {case}')
            if expected_source=='manual' and not math.isclose(actual['alpha'],coulomb['alpha'],rel_tol=1e-5):
                raise ValueError(f'Requested/actual alpha mismatch: {case}')
            expected=cfg['execution']['diagnostics']['fixed_samples']
            if r['n_samples']!=expected:raise ValueError(f'Incomplete samples: {case}: {r["n_samples"]} != {expected}')
            r.update(case=str(case),parameters=cfg['hyper_parameters'],solver_settings=actual,reference_converged=converged,
                measured_pppm_1e4_relative_rms=target,reference_1e6_vs_1e7_relative_rms=delta,
                reference_error_fraction=a.reference_error_fraction,reference_absolute_rms=ref_error,
                reference_allowed_absolute_rms=threshold,accuracy_status='REFERENCE_ADEQUATE' if converged else 'REFERENCE_LIMITED')
            (case/'force_statistics.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
            records.append(r)
    if not records:raise ValueError('No complete force experiments found')
    a.output.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','neighbor','coulomb','P','p','rs','rc','alpha_actual','alpha_source','rms','relative_rms','bias','reference_converged'])
        for r in records:
            h=r['parameters'];n=h['neighbor'];c=h['coulomb'];actual=r['solver_settings']
            w.writerow([r['case'],n['type'],c['type'],c['coulomb_sample_num'],n['neighbor_sample_num'],n['cut_off'],n['r_core'],actual['alpha'],actual['alpha_source'],r['rms_force_error'],r['relative_rms_force_error'],r['rms_bias'],r['reference_converged']])
    print(f'{len(records)} cases; RMS characterization only; no equal-accuracy performance gate')

if __name__=='__main__':main()
