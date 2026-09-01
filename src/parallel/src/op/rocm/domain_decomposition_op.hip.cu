#include "common/rbmd_define.h"
#include "domain_decomposition_op.h"

__device__ inline bool IsInRegion(const rbmd::Real* d_start_region,
                                  const rbmd::Real* d_end_region,
                                  const rbmd::Real px, const rbmd::Real py,
                                  const rbmd::Real pz) {
  return (px >= d_start_region[0] && px < d_end_region[0] &&
          py >= d_start_region[1] && py < d_end_region[1] &&
          pz >= d_start_region[2] && pz < d_end_region[2]);
}

namespace op {
__global__ void HandleDomainLeavingAtomsOneDim(
    rbmd::Real* d_start_region, rbmd::Real* d_end_region, rbmd::Real* px,
    rbmd::Real* py, rbmd::Real* pz, rbmd::Real* pos_to_modify, Box d_box,
    rbmd::Id dim, rbmd::Real shift, rbmd::Id total_atoms_num) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_atoms_num) {
    return;
  }
  if (IsInRegion(d_start_region, d_end_region, px[idx], py[idx], pz[idx])) {
    pos_to_modify[idx] = pos_to_modify[idx] + shift;
    if (shift < 0) {
      if (pos_to_modify[idx] <= d_box._coord_min[dim]) {
        pos_to_modify[idx] = d_box._coord_min[dim];
      }
    } else {
      if (pos_to_modify[idx] >= d_box._coord_max[dim]) {
        rbmd::Real r = d_box._coord_max[dim];
        pos_to_modify[idx] = NEXTAFTER(r, r - 1.0);
      }
    }
  }
}

__global__ void PopulateHalo(rbmd::Real* d_start_region,
                             rbmd::Real* d_end_region, rbmd::Real* px,
                             rbmd::Real* py, rbmd::Real* pz, Box d_box,
                             rbmd::Real* halo_px, rbmd::Real* halo_py,
                             rbmd::Real* halo_pz, unsigned int* counter,
                             unsigned max_halo_size, unsigned dim,
                             rbmd::Real shift, rbmd::Id native_atoms_num,
                             // 拓扑相关参数
                             rbmd::Id* atoms_id, rbmd::Id* atoms_type,
                             rbmd::Id* halo_atoms_id, rbmd::Id* halo_atoms_type) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= native_atoms_num) {
    return;
  }

  // Check if the native particle is in the source region for halo generation.
  if (IsInRegion(d_start_region, d_end_region, px[idx], py[idx], pz[idx])) {
    // Atomically get a unique index to write the new halo particle.
    unsigned int write_idx = atomicAdd(counter, 1);

    // Ensure we don't write past the allocated memory for halos.
    if (write_idx < max_halo_size) {
      // Create a copy of the particle and apply the periodic shift
      // to the specified dimension 'dim'. This correctly places the
      // ghost particle in the halo region without incorrect clamping.
      if (dim == 0) {
        halo_px[write_idx] = px[idx] + shift;
        halo_py[write_idx] = py[idx];
        halo_pz[write_idx] = pz[idx];
      } else if (dim == 1) {
        halo_px[write_idx] = px[idx];
        halo_py[write_idx] = py[idx] + shift;
        halo_pz[write_idx] = pz[idx];
      } else if (dim == 2) {
        halo_px[write_idx] = px[idx];
        halo_py[write_idx] = py[idx];
        halo_pz[write_idx] = pz[idx] + shift;
      }

      // 复制 atom_id 和 atom_type 用于拓扑力计算（bond/angle/dihedral）
      if (atoms_id != nullptr && halo_atoms_id != nullptr) {
        halo_atoms_id[write_idx] = atoms_id[idx];
      }
      if (atoms_type != nullptr && halo_atoms_type != nullptr) {
        halo_atoms_type[write_idx] = atoms_type[idx];
      }
    }
  }
}


__global__ void MarkLeavingAtomsInRegion(rbmd::Real* px, rbmd::Real* py,
                                         rbmd::Real* pz,
                                         PositionInfo* d_position_infos,
                                         
                                         rbmd::Id total_atoms_num,
                                         int* leaving_flags) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_atoms_num) {
    return;
  }
  int pos_leaving = 0;
  if (IsInRegion(d_position_infos->_leavingLow,
                 d_position_infos->_leavingHigh, px[idx], py[idx],
                 pz[idx])) {
    pos_leaving++;
                 }

  leaving_flags[idx] = (pos_leaving > 0) ? 1 : 0;
}


__global__ void ProcessLeavingAtom(rbmd::Real* p_px, rbmd::Real* p_py,
                                   rbmd::Real* p_pz,
                                   const PositionInfo* d_position_infos,
                                   rbmd::Id total_leaving, Box d_domain) { // d_domain 是全局盒子
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_leaving) {
    return;
  }
  rbmd::Real rx = p_px[idx];
  rbmd::Real ry = p_py[idx];
  rbmd::Real rz = p_pz[idx];
  if (IsInRegion(d_position_infos->_leavingLow,
                 d_position_infos->_leavingHigh, rx, ry, rz)) {

    // 1. 应用位移 (Apply shift)
    rx = rx + d_position_infos->_shift[0];
    ry = ry + d_position_infos->_shift[1];
    rz = rz + d_position_infos->_shift[2];

    // --- BEGIN FIX ---
    // 移植 CPU 版本的边界钳位 (clamping) 逻辑
    // [参考 CommunicationPartner.cpp, line 611-628][cite_start]// 假设 d_domain._coord_min[dim] 总是 0.0 (根据 localbox_info.txt) [cite: 1]

    // 维度 0 (X)
    if (d_position_infos->_shift[0] < 0) {
      if (rx < d_domain._coord_min[0]) {
        rbmd::Real r = d_domain._coord_min[0];
        rx = NEXTAFTER(r, r + 1.0f);
      }
    } else if (d_position_infos->_shift[0] > 0) {
      if (rx >= d_domain._coord_max[0]) {
        rbmd::Real r = d_domain._coord_max[0];
        rx = NEXTAFTER(r, r - 1.0f);
      }
    }

    // 维度 1 (Y)
    if (d_position_infos->_shift[1] < 0) {
      if (ry < d_domain._coord_min[1]) {
        rbmd::Real r = d_domain._coord_min[1];
        ry = NEXTAFTER(r, r + 1.0f);
      }
    } else if (d_position_infos->_shift[1] > 0) {
      if (ry >= d_domain._coord_max[1]) {
        rbmd::Real r = d_domain._coord_max[1];
        ry = NEXTAFTER(r, r - 1.0f);
      }
    }

    // 维度 2 (Z)
    if (d_position_infos->_shift[2] < 0) {
      if (rz < d_domain._coord_min[2]) {
        rbmd::Real r = d_domain._coord_min[2];
        rz = NEXTAFTER(r, r + 1.0f);
      }
    } else if (d_position_infos->_shift[2] > 0) {
      if (rz >= d_domain._coord_max[2]) {
        rbmd::Real r = d_domain._coord_max[2];
        rz = NEXTAFTER(r, r - 1.0f);
      }
    }

    // 2. 写回修复后的坐标
    p_px[idx] = rx;
    p_py[idx] = ry;
    p_pz[idx] = rz;
    // --- END FIX ---
                 }
}

__global__ void MarkHaloAtomsInRegion(rbmd::Real* px, rbmd::Real* py,
                                      rbmd::Real* pz,
                                      PositionInfo* d_position_infos,
                                      rbmd::Id mark_atoms_num,
                                      int* halo_flags,
                                      int* halo_owner,
                                      int info_index) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= mark_atoms_num) {
    return;
  }
  if (IsInRegion(d_position_infos->_copiesLow,
                 d_position_infos->_copiesHigh, px[idx], py[idx],
                 pz[idx])) {
    halo_flags[idx] = 1;
    if (halo_owner[idx] < 0) {
      halo_owner[idx] = info_index;
    }
  }
}

__global__ void ProcessHaloAtom(rbmd::Real* p_px, rbmd::Real* p_py,
                                rbmd::Real* p_pz,
                                const PositionInfo* d_position_infos,
                                rbmd::Id total_halo, Box d_domain,
                                const int* halo_owner_idx, int owner_idx) {
  (void)d_domain;
  // 获取全局线程ID
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_halo) {
    return;
  }

  if (halo_owner_idx[idx] != owner_idx) {
    return;
  }

  // 读取要处理的原子的当前坐标
  rbmd::Real rx = p_px[idx];
  rbmd::Real ry = p_py[idx];
  rbmd::Real rz = p_pz[idx];

  // 遍历所有可能的位置信息（通常对应不同的邻居方向）
  // 检查原子是否位于当前定义的"copies区域"（halo源区域）
  if (IsInRegion(d_position_infos->_copiesLow,
                 d_position_infos->_copiesHigh, rx, ry, rz)) {
    // Halo ghost 应落在扩展域里，和 PopulateHalo() 保持一致：
    // 这里只做周期平移，不对平移后的坐标做额外 clamping。
    p_px[idx] = rx + d_position_infos->_shift[0];
    p_py[idx] = ry + d_position_infos->_shift[1];
    p_pz[idx] = rz + d_position_infos->_shift[2];
  }

}


__device__ bool IsInRegionDevice(const rbmd::Real* sr, const rbmd::Real* er,
                                 rbmd::Real x, rbmd::Real y, rbmd::Real z) {
  return (x >= sr[0] && x < er[0] &&
          y >= sr[1] && y < er[1] &&
          z >= sr[2] && z < er[2]);
}

__global__ void CountHaloCandidates(
    const rbmd::Real* d_start_region, // 设备端内存，定义搜索区域的起始边界
    const rbmd::Real* d_end_region, // 设备端内存，定义搜索区域的结束边界
    const rbmd::Real* px, // 所有本地原子的x坐标数组 (设备端)
    const rbmd::Real* py, // 所有本地原子的y坐标数组 (设备端)
    const rbmd::Real* pz, // 所有本地原子的z坐标数组 (设备端)
    rbmd::Id native_atoms_num, // 本地域中原子的总数
    unsigned int* d_global_count) // 全局计数器 (单个 unsigned int, 在调用前应初始化为0)
{
  // 声明共享内存用于存储当前线程块的局部计数总和
  // __shared__ 变量对于块内的所有线程是可见的
  __shared__ unsigned int s_block_partial_count;

  // 由块内的第一个线程初始化该块的共享内存计数器
  if (threadIdx.x == 0) {
    s_block_partial_count = 0;
  }

  // 同步块内的所有线程，确保 s_block_partial_count 已被初始化
  // 然后才能安全地被其他线程访问
  __syncthreads();

  // 计算当前线程的全局索引
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 检查线程索引是否越界 (处理粒子总数不是线程块大小整数倍的情况)
  if (idx < native_atoms_num) {
    // 检查当前线程处理的粒子是否在指定区域内
    if (IsInRegionDevice(d_start_region, d_end_region, px[idx], py[idx],
                         pz[idx])) {
      // 如果粒子在区域内，原子地增加本块的共享内存计数器
      // 使用原子操作确保即使块内有多个线程同时更新 s_block_partial_count 也不会出错
      atomicAdd(&s_block_partial_count, 1);
                         }
  }

  // 再次同步块内的所有线程
  // 确保所有线程都已经完成了对 s_block_partial_count 的更新
  __syncthreads();

  // 由块内的第一个线程将该块的局部计数总和原子地加到全局计数器上
  if (threadIdx.x == 0) {
    // 只有当这个块确实有符合条件的粒子时才执行全局原子操作
    if (s_block_partial_count > 0) {
      atomicAdd(d_global_count, s_block_partial_count);
    }
  }
}


void op::HandleDomainLeavingAtomsOneDimOp<device::DEVICE_GPU>::operator()(
    rbmd::Real* d_start_region, rbmd::Real* d_end_region, rbmd::Real* px,
    rbmd::Real* py, rbmd::Real* pz, rbmd::Real* pos_to_modify, Box d_box,
    rbmd::Id dim, rbmd::Real shift, rbmd::Id total_atoms_num) {
  unsigned int blocks_per_grid =
      (total_atoms_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  HandleDomainLeavingAtomsOneDim<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      d_start_region, d_end_region, px, py, pz, pos_to_modify, d_box, dim,
      shift, total_atoms_num);
}

void op::PopulateHaloOp<device::DEVICE_GPU>::operator()(
    rbmd::Real* d_start_region, rbmd::Real* d_end_region, rbmd::Real* px,
    rbmd::Real* py, rbmd::Real* pz, Box d_box, rbmd::Real* halo_px,
    rbmd::Real* halo_py, rbmd::Real* halo_pz, unsigned int* counter,
    unsigned max_halo_size, unsigned dim, rbmd::Real shift,
    rbmd::Id native_atoms_num,
    rbmd::Id* atoms_id, rbmd::Id* atoms_type,
    rbmd::Id* halo_atoms_id, rbmd::Id* halo_atoms_type) {
  unsigned int blocks_per_grid =
      (native_atoms_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  PopulateHalo<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      d_start_region, d_end_region, px, py, pz, d_box, halo_px, halo_py,
      halo_pz, counter, max_halo_size, dim, shift, native_atoms_num,
      atoms_id, atoms_type, halo_atoms_id, halo_atoms_type);
}

void op::MarkLeavingAtomsInRegionOp<device::DEVICE_GPU>::operator()(
    rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    PositionInfo* d_position_infos, 
    rbmd::Id native_atoms_num, int* leaving_flags) {
  unsigned int blocks_per_grid =
      (native_atoms_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  MarkLeavingAtomsInRegion<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      px, py, pz, d_position_infos, native_atoms_num,
      leaving_flags);
}


void op::ProcessLeavingAtomOp<device::DEVICE_GPU>::operator()(
    rbmd::Real* p_px, rbmd::Real* p_py,
    rbmd::Real* p_pz, PositionInfo* d_position_infos,
     rbmd::Id total_leaving, Box d_domain) {
  unsigned int blocks_per_grid = (total_leaving + BLOCK_SIZE - 1) / BLOCK_SIZE;
  ProcessLeavingAtom<<<blocks_per_grid, BLOCK_SIZE>>>(
      p_px, p_py, p_pz, d_position_infos, total_leaving,
      d_domain);
}

void op::MarkHaloAtomsInRegionOp<device::DEVICE_GPU>::operator()(
    rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
    PositionInfo* d_position_infos,
    rbmd::Id mark_atoms_num, int* halo_flags,
    int* halo_owner, int info_index) {
  unsigned int blocks_per_grid =
      (mark_atoms_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  MarkHaloAtomsInRegion<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      px, py, pz, d_position_infos, mark_atoms_num,
      halo_flags, halo_owner, info_index);
}

void op::ProcessHaloAtomOp<device::DEVICE_GPU>::operator()(
    rbmd::Real* p_px, rbmd::Real* p_py,
    rbmd::Real* p_pz, PositionInfo* d_position_infos,
    rbmd::Id total_halo, Box d_domain,
    const int* halo_owner, int owner_idx) {
  unsigned int blocks_per_grid = (total_halo + BLOCK_SIZE - 1) / BLOCK_SIZE;
  ProcessHaloAtom<<<blocks_per_grid, BLOCK_SIZE>>>(
      p_px, p_py, p_pz, d_position_infos, total_halo,
      d_domain, halo_owner, owner_idx);
}


void op::CountHaloCandidatesOp<device::DEVICE_GPU>::operator()(
    const rbmd::Real* d_start_region, // 设备端内存，定义搜索区域的起始边界
    const rbmd::Real* d_end_region, // 设备端内存，定义搜索区域的结束边界
    const rbmd::Real* px, // 所有本地原子的x坐标数组 (设备端)
    const rbmd::Real* py, // 所有本地原子的y坐标数组 (设备端)
    const rbmd::Real* pz, // 所有本地原子的z坐标数组 (设备端)
    rbmd::Id native_atoms_num, // 本地域中原子的总数
    unsigned int* d_global_count) {
  int threads_per_block = 256; // 或者您选择的合适值
  int blocks = (native_atoms_num + threads_per_block - 1) / threads_per_block;

  CountHaloCandidates<<<blocks, threads_per_block>>>(
      d_start_region, d_end_region,
      px, py, pz,
      native_atoms_num,
      d_global_count
      );
}
}
// namespace op

// namespace op
