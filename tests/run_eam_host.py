#!/usr/bin/env python3
"""Compile production EAM kernels and wrappers, check arithmetic on one-thread blocks.

GPU launch/scheduling and MPI are not emulated. Scalar exchange is explicitly
performed by the test between owned-atom density and force evaluation.
"""
from pathlib import Path
import os, re, shutil, subprocess, tempfile
ROOT = Path(__file__).resolve().parents[1]

def definition(source, marker):
    start=source.index(marker); body=source.index('{',start); depth=1; end=body+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]

def main():
    source=(ROOT/'src/force/src/eam_op/rocm/eam_op.hip.cu').read_text()
    decl=(ROOT/'src/force/src/eam_op/eam_op.h').read_text()
    kernels=source[source.index('//device'):source.index('void ComputeFpVerlet<')]
    ops=[]
    names=re.findall(r'void (\w+)<device::DEVICE_GPU>::operator\(\)',source)
    for name in names:
        declaration=definition(decl,'struct '+name+'<device::DEVICE_GPU>')
        impl=definition(source,'void '+name+'<device::DEVICE_GPU>::operator()')
        ops.append('template<class> struct '+name+';\ntemplate<>\n'+declaration+';\n'+re.sub(r'<<<.*?>>>','',impl,flags=re.S))
    traversal=definition(decl,'struct EAMDirectTraversalData')+';'
    parameters=definition((ROOT/'src/force/include/eam.h').read_text(),'struct EAMParameters')+';'
    halo=(ROOT/'src/parallel/src/domain_decomposition/indirect_neighbour_communication_scheme.cpp').read_text()
    halo_structs='namespace halo_test { constexpr int kMaxForwardFields=12;\n'+definition(halo,'struct PackForwardFields')+';\n'+definition(halo,'struct ApplyForwardFields')+';\n}'
    correction=definition((ROOT/'src/force/src/eam.cpp').read_text(),'void EAM::ApplyRblForceCorrection(')
    cxx=os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    with tempfile.TemporaryDirectory(prefix='rbmd-eam-host-') as d:
        temp=Path(d)
        (temp/'production_eam.h').write_text(parameters+'\nnamespace op {\n'+traversal+'\n'+kernels+'\n'+'\n'.join(ops)+'\n}')
        (temp/'production_halo.h').write_text(halo_structs)
        (temp/'production_rbl_correction.h').write_text(correction)
        for precision in (0,1):
            exe=str(temp/('eam-'+str(precision)))
            subprocess.run([cxx,'-std=c++17','-O1','-g','-fsanitize=address,undefined',
                '-fno-omit-frame-pointer','-DUSE_64BIT_IDS=1','-DUSE_DOUBLE='+str(precision),
                '-I'+str(ROOT/'src'),'-I'+d,str(ROOT/'tests/eam_host.cpp'),'-o',exe],check=True)
            subprocess.run([exe],check=True)
if __name__=='__main__': main()
