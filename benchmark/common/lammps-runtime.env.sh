#!/bin/bash
# Match the rbm26 binary's compiler, HIP and MPI libraries.
export PATH=/public/software/compiler/dtk-26.04/bin:/public/software/compiler/dtk-26.04/llvm/bin:/public/software/compiler/gcc-12.2.0/bin:/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/comm_libs/openmpi/5.0.3-gcc-shca/bin:$PATH
export LD_LIBRARY_PATH=/public/software/compiler/dtk-26.04/lib:/public/software/compiler/dtk-26.04/lib64:/public/software/compiler/gcc-12.2.0/lib64:/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/comm_libs/openmpi/5.0.3-gcc-shca/lib:${LD_LIBRARY_PATH:-}
export HIP_PATH=/public/software/compiler/dtk-26.04
export ROCM_PATH=$HIP_PATH
