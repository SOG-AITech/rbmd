#!/usr/bin/env python3
"""Launch one allocated case, retaining exit status and reproducibility metadata."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from gpu_binding import validate

def validate_layout(ranks,nodes,expected=None,reference=False):
    if not reference and ranks != 4*nodes and (ranks,nodes) not in ((1,1),(2,1)):
        raise ValueError('Use four DCUs per node, with one/two-device single-node points allowed')
    if expected and (ranks,nodes,ranks//nodes)!=(expected['gpus'],expected['nodes'],expected['gpus_per_node']):
        raise ValueError('Allocation does not match scaling case layout')

def main():
    root=Path(__file__).resolve().parents[1]; case=Path(sys.argv[1])
    ranks=int(os.environ['SLURM_NTASKS']); nodes=int(os.environ['SLURM_JOB_NUM_NODES'])
    if ranks<1 or nodes<1 or ranks%nodes: raise SystemExit('Invalid MPI layout')
    reference=os.environ['NB_ENGINE']=='lammps' and os.environ.get('NB_LAMMPS_ROLE')=='reference'
    meta=json.loads((case/'case.json').read_text()) if (case/'case.json').is_file() else {}
    validate_layout(ranks,nodes,meta.get('layout'),reference)
    exe=Path(os.environ['RBMD_BIN' if os.environ['NB_ENGINE']=='rbmd' else ('LAMMPS_REFERENCE_BIN' if reference else 'LAMMPS_BIN')])
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
        return h.hexdigest()
    cmd=['mpirun','-np',str(ranks),'--map-by',f'ppr:{ranks//nodes}:node','--bind-to','none','bash',str(root/'common/rank.sh')]
    def capture(command):
        try:
            p=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=20)
            return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except (OSError,subprocess.TimeoutExpired) as e: return dict(command=command,error=str(e))
    runtime={name:capture(command) for name,command in {'mpi':['mpirun','--version'],'hip':['hipconfig','--version'],'libraries':['ldd',str(exe)]}.items()}
    receipt={'runtime':runtime,'lammps_role':os.environ.get('NB_LAMMPS_ROLE'),'communication_requested':'device_buffers' if not reference else 'host_buffers','command':cmd,'binary':str(exe),'binary_sha256':sha(exe),'job':os.environ['SLURM_JOB_ID'],
        'nodes':nodes,'ranks':ranks,'gpus_per_node':0 if reference else ranks//nodes,'input_sha256':sha(case/('rbmd.json' if os.environ['NB_ENGINE']=='rbmd' else 'run.in')),
        'profile':os.environ.get('NB_PROFILE','off'),
        'environment':{k:v for k,v in os.environ.items() if k in ('PATH','LD_LIBRARY_PATH','ROCM_PATH','NB_RUNTIME','NB_LAMMPS_RUNTIME') or k.startswith(('UCX_','OMPI_MCA_','HIP_','RBMD_','SLURM_JOB_'))},'start_unix':time.time()}
    start=time.monotonic()
    with (case/'stdout.log').open('x') as out:
        result=subprocess.run(cmd,cwd=case,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
    receipt.update(exit_code=result.returncode,process_wall_s=time.monotonic()-start)
    if not reference:
        try:
            receipt['gpu_bindings']=validate([json.loads(p.read_text()) for p in case.glob('gpu_binding.rank*.json')],ranks,nodes)
        except ValueError as e:
            receipt['gpu_binding_error']=str(e)
            receipt['exit_code']=result.returncode or 2
    with (case/'run_receipt.json').open('x') as out: json.dump(receipt,out,indent=2)
    raise SystemExit(receipt['exit_code'])

if __name__=='__main__': main()
