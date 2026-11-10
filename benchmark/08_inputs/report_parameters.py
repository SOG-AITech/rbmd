#!/usr/bin/env python3
"""Collect configured and measured parameters; missing measurements remain explicit."""
import argparse,csv,json,math,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from read_kspace import settings
from audit_sources import digest,header

def charges(path,style):
    inside=False;count=0;total=0.;square=0.;values=set()
    with path.open() as f:
        for line in f:
            text=line.split('#',1)[0].strip()
            if text=='Atoms':inside=True;continue
            if not inside or not text:continue
            fields=text.split()
            if not fields[0].isdigit():break
            q=float(fields[3 if style=='full' else 2])
            if not math.isfinite(q):raise ValueError('Nonfinite charge')
            count+=1;total+=q;square+=q*q;values.add(q)
    if count!=header(path):raise ValueError('Atoms/charge count mismatch')
    return dict(base_atoms=count,net_charge=total,sum_q2=square,charges=sorted(values))

def sampled_counts(case):
    files=sorted(case.glob('revision.rbl_samples.rank*.csv'))
    if not files:return None
    receipt=json.loads((case/'run_receipt.json').read_text())
    if len(files)!=receipt['ranks']:raise ValueError('Missing RBL sample ranks')
    per_rank=[]
    for path in files:
        rows={}
        with path.open() as f:
            for row in csv.DictReader(f):
                # A force evaluation can rebuild twice; retain its final list.
                sample=int(row['sample']);r={k:float(row[k]) for k in ('n_owned','sum','sum_square','min','max')}
                if sample<1 or any(not math.isfinite(v) or v<0 for v in r.values()):raise ValueError('Invalid sample counts')
                rows[sample]=r
        per_rank.append(rows)
    steps=set(per_rank[0]);cfg=json.loads((case/'rbmd.json').read_text())
    expected=cfg['execution']['diagnostics']['fixed_samples']
    if steps!=set(range(1,expected+1)) or any(set(rows)!=steps for rows in per_rank):raise ValueError('Incomplete RBL sample epochs')
    total=sum(row['n_owned'] for rows in per_rank for row in rows.values())
    if total<=0:raise ValueError('No owned samples')
    per_frame=[sum(rows[i]['n_owned'] for rows in per_rank) for i in steps]
    pre=json.loads((case/'preflight.json').read_text())
    if any(n!=pre['n_atoms'] for n in per_frame):raise ValueError('RBL owned count differs from full N')
    mean=sum(row['sum'] for rows in per_rank for row in rows.values())/total
    second=sum(row['sum_square'] for rows in per_rank for row in rows.values())/total
    return dict(mean=mean,std=math.sqrt(max(0.,second-mean*mean)),min=min(row['min'] for rows in per_rank for row in rows.values()),max=max(row['max'] for rows in per_rank for row in rows.values()),samples=len(steps),measurement='owned-atom materialized shell counts before pair force; includes selected special pairs even if their force is scaled/excluded')

def record(cfg,meta,case=None,cache=None):
    h=cfg['hyper_parameters'];n=h['neighbor'];c=h['coulomb'];e=cfg['execution'];d=cfg['init_configuration']['read_data'];data=Path(d['file'])
    cache={} if cache is None else cache
    if str(data) not in cache:cache[str(data)]=(charges(data,d['atom_style']),digest(data))
    charge,sha=cache[str(data)];rep=d.get('replicate',[1,1,1]);N=charge['base_atoms']*math.prod(rep)
    log=(case/'stdout.log').read_text(errors='replace') if case and (case/'stdout.log').exists() else ''
    actual=settings(log);receipt=json.loads((case/'run_receipt.json').read_text()) if case and (case/'run_receipt.json').exists() else {}
    effective=sampled_counts(case) if case else None
    bjerrum=dict(charge,convention='q in data units; LJ unit-charge Bjerrum length is 1/(dielectric*T)')
    if d['unit'].upper()=='LJ':bjerrum.update(bjerrum_length_sigma=1/e['temperature'][0],dielectric=1,temperature=e['temperature'][0])
    fft=re.findall(r'KOKKOS FFT library\s*=\s*(\S+)',log)
    if not fft:fft=re.findall(r'FFT library\s*=\s*(\S+)',log)
    if not fft:fft=re.findall(r'using (?:single|double) precision (\S+)',log,re.I)
    return dict(case=str(case) if case else 'configured:'+meta['system'],system=meta['system'],exact_N=N,data_sha256=sha,
        construction=d['file'],equilibration=meta.get('equilibration_status','UNVERIFIED'),replicate=rep,
        full_box_sampling='see evidence/20261017/source-input-audit.json; verify generated runtime evidence',precision='double-reference-only' if meta.get('lammps_role')=='reference' else 'single',
        rs=n['cut_off'],rc=n['r_core'],P=c['coulomb_sample_num'],p_nominal=n['neighbor_sample_num'],
        p_effective=effective['mean'] if effective else 'NOT_MEASURED',p_effective_measurement=effective or 'accuracy-only force samples required',
        alpha_actual=actual.get('alpha','NOT_MEASURED'),dt=e['timestep'],temperature=e['temperature'][:2],thermostat_damping=e['temperature'][2],thermostat_chain=3,
        special_bonds=h.get('extend',{}).get('special_bonds',[0,0,0]),lj_mixing=h['force_field']['mixing_rule'],charge_or_bjerrum_length=bjerrum,
        pppm_target=(re.findall(r'kspace_style pppm (\S+)',(case/'run.in').read_text())[0] if case and (case/'run.in').exists() else 'NOT_APPLICABLE'),
        pppm_actual_rms='see force_statistics.json',pppm_order=actual.get('pppm_order','NOT_MEASURED'),pppm_grid=actual.get('pppm_grid','NOT_MEASURED'),gewald=actual.get('gewald','NOT_MEASURED'),
        fft_backend=fft[-1] if fft else 'NOT_MEASURED',nodes=receipt.get('nodes','NOT_RUN'),gpus_per_node=receipt.get('gpus_per_node','NOT_RUN'),mpi_ucx=receipt.get('runtime','NOT_RUN'),
        seed=c.get('random_seed','UNSET'),force_error_evidence=str(case/'force_statistics.json') if case and (case/'force_statistics.json').exists() else 'NOT_RUN',timing_evidence=str(case/'run_receipt.json') if receipt else 'NOT_RUN',status=('RUN_COMPLETED' if receipt.get('exit_code')==0 else 'RUN_FAILED') if receipt else 'CONFIGURED_NOT_RUN')

def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--systems',type=Path);group.add_argument('--cases',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cache={};rows=[]
    if a.systems:
        for f in sorted(a.systems.glob('*.json')):
            s=json.loads(f.read_text());r=record(s['rbmd'],dict(system=s['name'],equilibration_status=s['equilibration_status']),cache=cache);r['case']=str(f);rows.append(r)
    else:
        for f in sorted(a.cases.rglob('case.json')):rows.append(record(json.loads(f.with_name('rbmd.json').read_text()),json.loads(f.read_text()),f.parent,cache))
    if not rows:raise ValueError('No cases/systems')
    with a.output.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    print(f'{len(rows)} parameter rows: {a.output}')
if __name__=='__main__':main()
