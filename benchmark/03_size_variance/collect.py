#!/usr/bin/env python3
"""Collect MJ2(a) force statistics, separately by algorithm, from a prepared size scan."""
import argparse,json,math
from pathlib import Path

METHODS={'rbe':('Verlet-List','RBE'),'combined':('RBL','RBE')}

def collect(root):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['experiment']!='MJ2(a)':raise ValueError('Expected MJ2(a) manifest')
    fixed=manifest['fixed_parameters'];inputs=manifest['inputs'];result={}
    if manifest.get('methods',list(METHODS))!=list(METHODS):raise ValueError('Expected RBE and RBE+RBL size scan')
    if len(inputs)<2 or len({e['N'] for e in inputs})!=len(inputs):raise ValueError('Need >=2 distinct N')
    for method,types in METHODS.items():
        rows=[]
        for entry in inputs:
            case=root/f"N{entry['N']}"/f"rs{fixed['rs']:g}"/method
            cfg=json.loads((case/'rbmd.json').read_text());s=json.loads((case/'force_statistics.json').read_text())
            receipt=json.loads((case/'run_receipt.json').read_text());pre=json.loads((case/'preflight.json').read_text())
            h=cfg['hyper_parameters'];n=h['neighbor'];c=h['coulomb'];d=cfg['init_configuration']['read_data']
            actual=dict(P=c['coulomb_sample_num'],p=n['neighbor_sample_num'],alpha=c.get('alpha'),rs=n['cut_off'],rc=n['r_core'],dt=cfg['execution']['timestep'])
            if actual!=fixed or (n['type'],c['type'])!=types or not c.get('debug_alpha'):raise ValueError('Algorithm/fixed parameters mismatch')
            if (d['file']!=entry['path'] or d.get('replicate',[1,1,1])!=[1,1,1] or
                pre.get('status')!='PASS' or pre.get('data_sha256')!=entry['sha256'] or pre.get('n_atoms')!=entry['N'] or receipt['exit_code']!=0):raise ValueError('Input provenance or completed-run check failed')
            observed=s['solver_settings']
            if observed.get('alpha_source')!='manual' or not math.isclose(observed['alpha'],fixed['alpha'],rel_tol=1e-5):raise ValueError('Actual alpha differs from fixed alpha')
            if s['parameters']!=h or s['n_atoms']!=entry['N'] or s['n_samples']!=manifest['samples']:raise ValueError('Statistics/configuration mismatch')
            if not s.get('reference_converged'):raise ValueError('Reference is limited; converge it before the size comparison')
            values={key:s[key] for key in ('mse','rms_force_error','rms_bias','variance_population','variance_unbiased')}
            if any(v is None or not math.isfinite(v) or v<0 for v in values.values()):raise ValueError('Invalid force statistics')
            rows.append(dict(N=entry['N'],samples=s['n_samples'],alpha_actual=observed['alpha'],**values,
                mse_per_N=s['mse']/entry['N'],variance_per_N=s['variance_unbiased']/entry['N']))
        rows.sort(key=lambda r:r['N'])
        slope=None
        if all(r['variance_unbiased']>0 for r in rows):
            x=[math.log(r['N']) for r in rows];y=[math.log(r['variance_unbiased']) for r in rows]
            xm=sum(x)/len(x);ym=sum(y)/len(y)
            slope=sum((u-xm)*(v-ym) for u,v in zip(x,y))/sum((u-xm)**2 for u in x)
        result[method]=dict(rows=rows,log_variance_vs_log_N_slope=slope)
    return dict(fixed_parameters=fixed,methods=result,
        definitions='mse = mean over atoms and batches of |F-Fref|^2; RMS=sqrt(mse). Population variance is about the batch mean; mse = squared bias + population variance. Unbiased variance uses M-1.',
        scope='Selected sizes only. No maximum-N extrapolation, heat flux or NVE. Batch samples at one configuration do not measure equilibrium-state uncertainty.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(collect(a.root),indent=2,allow_nan=False)+'\n');print(a.output)
if __name__=='__main__':main()
