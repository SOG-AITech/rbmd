#!/usr/bin/env python3
"""Record visible physical GPUs before MD; preserve the scheduler's visibility masks."""
import ctypes,json,os,socket
from pathlib import Path

def validate(records,ranks,nodes):
    if len(records)!=ranks or {r['rank'] for r in records}!=set(range(ranks)):
        raise ValueError('Incomplete GPU binding records')
    hosts={r['host'] for r in records}
    if len(hosts)!=nodes:raise ValueError('GPU binding node count mismatch')
    for host in hosts:
        rows=[r for r in records if r['host']==host]
        if len(rows)!=ranks//nodes or len({r['pci_bus_id'] for r in rows})!=len(rows):
            raise ValueError('MPI ranks share a physical GPU or have an uneven layout: '+host)
    return records

def main():
    rank=int(os.environ['OMPI_COMM_WORLD_RANK']);local=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    size=int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    hip=ctypes.CDLL('libamdhip64.so')
    hip.hipGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
    count=ctypes.c_int()
    if hip.hipGetDeviceCount(ctypes.byref(count))!=0 or count.value!=size or not 0<=local<size:
        raise RuntimeError('Expected exactly one visible GPU per local MPI rank; inspect Slurm/HIP/ROCR masks')
    bus=ctypes.create_string_buffer(64)
    if hip.hipDeviceGetPCIBusId(bus,len(bus),local)!=0:raise RuntimeError('Cannot query HIP PCI bus ID')
    record=dict(rank=rank,local_rank=local,local_size=size,host=socket.gethostname(),visible_count=count.value,
        selected_ordinal=local,pci_bus_id=bus.value.decode(),
        visibility={key:os.environ.get(key) for key in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','SLURM_JOB_GPUS','SLURM_STEP_GPUS')})
    with Path(f'gpu_binding.rank{rank}.json').open('x') as f:json.dump(record,f,indent=2)
if __name__=='__main__':main()
