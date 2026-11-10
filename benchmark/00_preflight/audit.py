#!/usr/bin/env python3
"""Fail before simulation when precision, files or layout are incompatible."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path);p.add_argument('--engine',choices=['rbmd','lammps']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    meta=json.loads((a.case/'case.json').read_text()) if a.case else {}
    reference=meta.get('lammps_role')=='reference'
    bins={'rbmd':Path(os.environ.get('RBMD_BIN',str(root/'software/rbmd-single/rbmd'))),
          'lammps':Path(os.environ.get('LAMMPS_BIN','/public/home/acymkxdx47/software/rbm26/build-hip-single/lmp'))}
    if reference: bins['lammps']=Path(os.environ.get('LAMMPS_REFERENCE_BIN',str(root/'software/lammps-reference-double/lmp')))
    errors=[]; report={'binaries':{},'errors':errors,'status':'NOT_RUN'}
    for name,binary in bins.items():
        if a.engine and name != a.engine: continue
        cache=binary.parent/'CMakeCache.txt'; entry={'path':str(binary),'exists':binary.is_file()}
        if not binary.is_file(): errors.append(f'{name} binary missing: {binary}')
        if cache.is_file():
            settings=dict(re.findall(r'^([A-Za-z0-9_]+):[^=\n]+=(.*)$',cache.read_text(),re.M))
            keys=('ENABLE_DOUBLE','USE_MPI','FFT','FFT_KOKKOS','FFT_SINGLE','KOKKOS_PREC','Kokkos_ENABLE_HIP','PKG_MOLECULE','PKG_RIGID','PKG_KOKKOS')
            entry['build']={k:settings.get(k) for k in keys}
            if name=='rbmd' and settings.get('ENABLE_DOUBLE')!='OFF': errors.append('RBMD requires ENABLE_DOUBLE=OFF for this single-precision benchmark')
            if name=='lammps' and not reference and settings.get('Kokkos_ENABLE_HIP')!='ON': errors.append('LAMMPS lacks HIP Kokkos')
            if name=='lammps':
                required={'FFT_SINGLE':'ON','KOKKOS_PREC':'single','FFT':'KISS','FFT_KOKKOS':'HIPFFT','PKG_MOLECULE':'ON','PKG_RIGID':'ON'}
                if reference: required={'FFT_SINGLE':'OFF','PKG_KOKKOS':'OFF','PKG_MOLECULE':'ON','PKG_RIGID':'ON'}
                for key,value in required.items():
                    if settings.get(key)!=value: errors.append(f'LAMMPS requires {key}={value}; found {settings.get(key)}')
        else: errors.append(f'Cannot verify build precision: {cache}')
        if name == 'rbmd' and binary.absolute() == Path('/public/home/acymkxdx47/ReviewRBMD/rbmd/build/rbmd'):
            authority = json.loads(Path('/public/home/acymkxdx47/ReviewRBMD/NewBenchmark_six_systems_20261017/control/CURRENT_RBL_IMPLEMENTATION.json').read_text())
            release_path = Path(authority['release']['path'])
            if hashlib.sha256(release_path.read_bytes()).hexdigest() != authority['release']['sha256']:
                errors.append('Current RBL release SHA mismatch')
            release = json.loads(release_path.read_text())
            actual_sha = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None
            entry['sha256'] = actual_sha
            entry['release'] = str(release_path)
            if (authority.get('status') != 'ACTIVE_VERIFIED' or
                    release['binary']['path'] != str(binary.absolute()) or
                    actual_sha != release['binary']['sha256']):
                errors.append('Current RBL binary path/SHA mismatch; requalify any rebuilt binary')
        report['binaries'][name]=entry
    if a.case:
        cfg=json.loads((a.case/'rbmd.json').read_text());data=Path(cfg['init_configuration']['read_data']['file'])
        report['data']=str(data)
        if not data.is_file(): errors.append('Input data missing')
        else:
            h=hashlib.sha256(); count=None
            with data.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):
                    h.update(b)
                    if count is None:
                        m=re.search(rb'\n\s*(\d+) atoms\s*\n',b)
                        if m: count=int(m.group(1))
            replicas=cfg['init_configuration']['read_data'].get('replicate',[1,1,1])
            if len(replicas)!=3 or any(type(v)!=int or v<1 for v in replicas): errors.append('Invalid replicate factors');replicas=[1,1,1]
            report.update(data_sha256=h.hexdigest(),base_atoms=count,n_atoms=count*math.prod(replicas) if count else None,replicate=replicas)
            if not count: errors.append('Cannot determine atom count')
        c=cfg['hyper_parameters']['coulomb']
        if 'alpha' in c or c.get('debug_alpha'):
            meta=json.loads((a.case/'case.json').read_text())
            diag=cfg['execution'].get('diagnostics',{})
            alpha=c.get('alpha')
            if not (meta.get('alpha_scan') and meta.get('engine')=='rbmd' and
                    diag.get('fixed_samples',0)>1 and (diag.get('forces') or diag.get('reference_force_file')) and
                    c.get('debug_alpha') is True and type(alpha) in (int,float) and
                    math.isfinite(alpha) and alpha>0):
                errors.append('Manual alpha only allowed in labeled fixed-configuration force scans')
        if c.get('rbe_low_modes'): errors.append('Unexpected low-mode estimator')
    report['status']='PASS' if not errors else 'BLOCKED'
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if errors: raise SystemExit(2)

if __name__=='__main__':main()
