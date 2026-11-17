#include "../common/rbmd_define.h"
#include "force/src/force_op/force_op.h"
#include "force_op.h"
#include "model/box.h"

namespace op {

//RBL: Fix RBL Force
__global__ void FixRBLForce(const rbmd::Id num_atoms,
                            const rbmd::Real corr_value_x,
                            const rbmd::Real corr_value_y,
                            const rbmd::Real corr_value_z, rbmd::Real* __restrict__ fx,
                            rbmd::Real* __restrict__ fy, rbmd::Real* __restrict__ fz) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_atoms) {
    fx[tid] -= corr_value_x;
    fy[tid] -= corr_value_y;
    fz[tid] -= corr_value_z;
  }
}


//////////////////////////////////////

//RBL: Fix LJForce
void FixRBLForceOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id num_atoms, const rbmd::Real corr_value_x,
    const rbmd::Real corr_value_y, const rbmd::Real corr_value_z,
    rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz) {
  unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(FixRBLForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      num_atoms, corr_value_x, corr_value_y, corr_value_z, fx, fy, fz));
}


}
