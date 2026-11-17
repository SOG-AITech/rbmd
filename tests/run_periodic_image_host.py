#!/usr/bin/env python3
"""Execute the production single-domain periodic kernel with a CPU launch shim."""
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'src/parallel/src/op/rocm/domain_decomposition_op.hip.cu').read_text()
inside=s[s.index('__device__ inline bool IsInRegion'):s.index('namespace op {')]
kernel=s[s.index('__global__ void HandleDomainLeavingAtomsOneDim'):s.index('__global__ void PopulateHalo')]
prefix=r'''
#include <cassert>
#include <cmath>
#include "common/types.h"
#define __device__
#define __global__
#define NEXTAFTER std::nextafter
#define ROUND std::round
struct Dim {int x;} blockIdx{0},blockDim{1},threadIdx{0};
struct Box {rbmd::Real _coord_min[3],_coord_max[3];};
'''
check=r'''
int main(){
 using R=rbmd::Real;using I=rbmd::Id;
 Box box{{0,0,0},{271,271,271}};
 // A tiny negative coordinate can round to exactly L after adding L in float.
 // The subsequent local wrap must undo the corresponding image increment.
 for(int dim=0;dim<3;++dim) for(int sign:{-1,1}) {
   R xyz[3]={4,5,6};xyz[dim]=sign<0?R(271):R(-.125);
   R lo[3]={-1,-1,-1},hi[3]={272,272,272};
   if(sign<0)lo[dim]=271;else hi[dim]=0;
   I image=sign<0?-1:2;R unwrapped=xyz[dim]+image*271;
   const R before=unwrapped;
   HandleDomainLeavingAtomsOneDim(lo,hi,&xyz[0],&xyz[1],&xyz[2],&xyz[dim],&image,&unwrapped,box,dim,R(sign*271),1);
   assert(image==(sign<0?0:1));
   assert(std::abs(unwrapped-before)<1e-4);
   assert(std::abs(xyz[dim]+image*271-before)<1e-4);
   assert(xyz[dim]>=0 && xyz[dim]<271);
 }
}
'''
# initializer_list is needed by the signed-direction test.
prefix='#include <initializer_list>\n'+prefix
with tempfile.TemporaryDirectory(prefix='periodic-image-') as temp:
 p=Path(temp);(p/'test.cpp').write_text(prefix+inside+kernel+check)
 for double in (0,1):
  for wide in (0,1):
   subprocess.run(['clang++','-std=c++17','-fsanitize=address,undefined','-fno-sanitize-recover=all',f'-DUSE_DOUBLE={double}',f'-DUSE_64BIT_IDS={wide}','-I'+str(ROOT/'src'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
   subprocess.run([str(p/'test')],check=True)
   print(f'PASS double={double} ids64={wide}: local periodic image continuity')
