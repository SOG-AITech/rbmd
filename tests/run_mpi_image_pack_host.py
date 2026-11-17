#!/usr/bin/env python3
"""Exercise the production SoA migration pack kernel with a CPU launch shim.
Checks signed periodic shifts, atom permutations, odd packet counts, buffer bounds
and float/double with 32/64-bit IDs. GPU/MPI integration is a separate test.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'src/parallel/src/op/rocm/gpu_comm_plan_op.hip.cu').read_text()
kernel = source[source.index('__device__ __forceinline__ rbmd::Real ApplyLeavingShiftAndClamp'):source.index('__global__ void PackGhostKernel')]
prefix = r'''
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>
#include "common/types.h"
#define __device__
#define __global__
#define __forceinline__ inline
#define NEXTAFTER std::nextafter
#define ROUND std::round
struct Dim { int x; } blockIdx{0}, blockDim{1}, threadIdx{0};
template<class T> T __ldg(const T* p) { return *p; }
'''
check = r'''
int main() {
 using I = rbmd::Id; using R = rbmd::Real;
 const int indices[4] = {2,0,1,3};
 const I ids[4]={10,20,30,40}, types[4]={1,2,3,4}, images[4]={2,-3,4,-1};
 const R x[4]={-1,17,4,10}, shift[4]={16,-16,0,0}, velocity[4]={1,2,3,4}, charge[4]={-1,1,-2,2};
 for (int count : {1,3,4}) for (bool has_charge : {false,true}) {
   const size_t ids_per_atom = 5 + (sizeof(R)>sizeof(I) ? 1 : 0);
   const size_t bytes = count*(ids_per_atom*sizeof(I)+(9+int(has_charge))*sizeof(R));
   std::vector<std::max_align_t> storage((bytes+64+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
   char* buffer=reinterpret_cast<char*>(storage.data());
   std::memset(buffer,0xab,bytes+64);
   for (int i=0;i<count;++i) {
     blockIdx.x=i;
     PackLeavingSoAKernel(indices,shift,shift,shift,count,ids,types,nullptr,
       x,x,x,velocity,velocity,velocity,charge,images,images,images,false,has_charge,
       buffer,0,0,0,16,16,16);
   }
   const I* packed_i=reinterpret_cast<I*>(buffer);
   const R* packed_r=reinterpret_cast<R*>(buffer+count*ids_per_atom*sizeof(I));
   assert(reinterpret_cast<std::size_t>(packed_r)%alignof(R)==0);
   for (int i=0;i<count;++i) {
     const int src=indices[i];
     assert(packed_i[i]==ids[src] && packed_i[count+i]==types[src]);
     for (int d=0;d<3;++d) {
       assert(packed_i[(2+d)*count+i]==images[src]-I(shift[src]/16));
       assert(packed_r[d*count+i]==x[src]+shift[src]);
       assert(packed_r[(3+d)*count+i]==velocity[src]);
       assert(packed_r[(6+int(has_charge)+d)*count+i]==x[src]+images[src]*16);
     }
     if(has_charge) assert(packed_r[6*count+i]==charge[src]);
   }
   for(size_t i=bytes;i<bytes+64;++i) assert(static_cast<unsigned char>(buffer[i])==0xab);
 }
}
'''
with tempfile.TemporaryDirectory(prefix='mpi-image-pack-') as temp:
    path = Path(temp); (path/'test.cpp').write_text(prefix+kernel+check)
    for double in (0,1):
        for wide in (0,1):
            subprocess.run(['clang++','-std=c++17','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                f'-DUSE_DOUBLE={double}',f'-DUSE_64BIT_IDS={wide}','-I'+str(ROOT/'src'),str(path/'test.cpp'),'-o',str(path/'test')],check=True)
            subprocess.run([str(path/'test')],check=True)
            print(f'PASS double={double} ids64={wide}: signed shifts, permutation, odd counts, bounds/alignment',flush=True)
