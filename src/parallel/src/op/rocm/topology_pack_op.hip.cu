#include "domain_decomposition_op.h"
#include "common/rbmd_define.h"
#include "domain_decomposition/topology_pack_op.h"
#include <cstring>
#include <assert.h>

// ubuf trick: 用于将整数编码进 double buffer（对齐 LAMMPS）
union ubuf {
  double d;
  int64_t i;
  
  // 构造函数需要 __host__ __device__ 修饰符以在 GPU kernel 中使用
  __host__ __device__ ubuf(double arg) : d(arg) {}
  __host__ __device__ ubuf(int64_t arg) : i(arg) {}
};

__device__ __forceinline__ int ClampCountToCapacity(int count, int capacity) {
  if (count < 0 || capacity <= 0) return 0;
  return (count > capacity) ? capacity : count;
}

/**
 * @brief 对单个维度应用 shift 后做边界钳位 (clamping)
 *
 * 与 ProcessLeavingAtom kernel 中的逻辑一致：
 * - shift < 0 且坐标 < box_min => clamp 到 nextafter(box_min, +∞)
 * - shift > 0 且坐标 >= box_max => clamp 到 nextafter(box_max, -∞)
 */
__device__ __forceinline__ rbmd::Real ApplyShiftAndClamp(
    rbmd::Real pos, rbmd::Real shift, rbmd::Real bmin, rbmd::Real bmax) {
  rbmd::Real r = pos + shift;
  if (shift < 0) {
    if (r < bmin) {
      r = NEXTAFTER(bmin, bmin + static_cast<rbmd::Real>(1.0));
    }
  } else if (shift > 0) {
    if (r >= bmax) {
      r = NEXTAFTER(bmax, bmax - static_cast<rbmd::Real>(1.0));
    }
  }
  return r;
}

/**
 * @brief GPU Kernel: 打包单个原子的拓扑数据到 exchange buffer
 *
 * 对齐 LAMMPS AtomVecFullKokkos_PackExchangeFunctor::operator()
 * 每个线程处理一个要发送的原子
 *
 * 新增：d_shift_x/y/z 为 per-send-atom 的位移数组（可为 nullptr），
 *       box_min/box_max 为全局盒子边界，用于 shift 后的边界钳位。
 *       这样做避免了在 pack 前原地修改 _d_px/py/pz 导致跨维度坐标污染。
 */
__global__ void PackExchangeTopologyKernel(
    int nsend, const int* sendlist, double* buf, int size_exchange,
    int nmax,
    // 基础属性
    const rbmd::Real* d_px, const rbmd::Real* d_py, const rbmd::Real* d_pz,
    const rbmd::Real* d_vx, const rbmd::Real* d_vy, const rbmd::Real* d_vz,
    const rbmd::Id* d_id, const rbmd::Id* d_type, const rbmd::Id* d_molecule,
    const rbmd::Real* d_charge,
    // Bond 数据
    const int* d_num_bond, const int* d_bond_type, const rbmd::Id* d_bond_atom,
    int bond_per_atom,
    // Angle 数据
    const int* d_num_angle, const int* d_angle_type,
    const rbmd::Id* d_angle_atom1, const rbmd::Id* d_angle_atom2,
    const rbmd::Id* d_angle_atom3, int angle_per_atom,
    // Dihedral 数据
    const int* d_num_dihedral, const int* d_dihedral_type,
    const rbmd::Id* d_dihedral_atom1, const rbmd::Id* d_dihedral_atom2,
    const rbmd::Id* d_dihedral_atom3, const rbmd::Id* d_dihedral_atom4,
    int dihedral_per_atom,
    // Improper 数据
    const int* d_num_improper, const int* d_improper_type,
    const rbmd::Id* d_improper_atom1, const rbmd::Id* d_improper_atom2,
    const rbmd::Id* d_improper_atom3, const rbmd::Id* d_improper_atom4,
    int improper_per_atom,
    // Special 数据
    const int* d_nspecial, const rbmd::Id* d_special, int maxspecial,
    // Position shift (per-send-atom, 可为 nullptr 表示无 shift)
    const rbmd::Real* d_shift_x, const rbmd::Real* d_shift_y,
    const rbmd::Real* d_shift_z,
    // 全局盒子边界（用于 clamping）
    rbmd::Real box_min_x, rbmd::Real box_min_y, rbmd::Real box_min_z,
    rbmd::Real box_max_x, rbmd::Real box_max_y, rbmd::Real box_max_z) {
  
  unsigned int mysend = blockIdx.x * blockDim.x + threadIdx.x;
  if (mysend >= nsend) return;

  // 获取要发送的原子索引
  const int i = sendlist[mysend];
#ifndef NDEBUG
  assert(i >= 0);
  assert(i < nmax);
#endif
  if (i < 0 || i >= nmax) {
    return;
  }

  // 计算 buffer 起始位置（扁平化存储）
  double* mybuf = buf + mysend * size_exchange;
  int m = 0;

  // [0] buffer 长度标记
  mybuf[m++] = static_cast<double>(size_exchange);

  // [1-3] 坐标（在 pack 内部应用 shift + clamping，不污染源数据）
  if (d_shift_x != nullptr) {
    mybuf[m++] = ApplyShiftAndClamp(d_px[i], d_shift_x[mysend],
                                     box_min_x, box_max_x);
    mybuf[m++] = ApplyShiftAndClamp(d_py[i], d_shift_y[mysend],
                                     box_min_y, box_max_y);
    mybuf[m++] = ApplyShiftAndClamp(d_pz[i], d_shift_z[mysend],
                                     box_min_z, box_max_z);
  } else {
    mybuf[m++] = d_px[i];
    mybuf[m++] = d_py[i];
    mybuf[m++] = d_pz[i];
  }

  // [4-6] 速度
  mybuf[m++] = d_vx[i];
  mybuf[m++] = d_vy[i];
  mybuf[m++] = d_vz[i];

	  // [7-10] id, type, molecule, charge
	  mybuf[m++] = ubuf(static_cast<int64_t>(d_id[i])).d;
	  mybuf[m++] = ubuf(static_cast<int64_t>(d_type[i])).d;
	  mybuf[m++] = ubuf(static_cast<int64_t>(d_molecule[i])).d;
	  mybuf[m++] = d_charge[i];

	  // [11+] Bond 数据：num_bond + bonds
	  int num_bond_i = ClampCountToCapacity(d_num_bond[i], bond_per_atom);
	  mybuf[m++] = ubuf(static_cast<int64_t>(num_bond_i)).d;
	  for (int k = 0; k < num_bond_i; k++) {
	    int idx = i * bond_per_atom + k;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_bond_type[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_bond_atom[idx])).d;
	  }

	  // Angle 数据：num_angle + angles
	  int num_angle_i = ClampCountToCapacity(d_num_angle[i], angle_per_atom);
	  mybuf[m++] = ubuf(static_cast<int64_t>(num_angle_i)).d;
	  for (int k = 0; k < num_angle_i; k++) {
	    int idx = i * angle_per_atom + k;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_angle_type[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_angle_atom1[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_angle_atom2[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_angle_atom3[idx])).d;
	  }

	  // Dihedral 数据：num_dihedral + dihedrals
	  int num_dihedral_i =
	      ClampCountToCapacity(d_num_dihedral[i], dihedral_per_atom);
	  mybuf[m++] = ubuf(static_cast<int64_t>(num_dihedral_i)).d;
	  for (int k = 0; k < num_dihedral_i; k++) {
	    int idx = i * dihedral_per_atom + k;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_dihedral_type[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_dihedral_atom1[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_dihedral_atom2[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_dihedral_atom3[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_dihedral_atom4[idx])).d;
	  }

	  // Improper 数据：num_improper + impropers
	  int num_improper_i =
	      ClampCountToCapacity(d_num_improper[i], improper_per_atom);
	  mybuf[m++] = ubuf(static_cast<int64_t>(num_improper_i)).d;
	  for (int k = 0; k < num_improper_i; k++) {
	    int idx = i * improper_per_atom + k;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_improper_type[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_improper_atom1[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_improper_atom2[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_improper_atom3[idx])).d;
	    mybuf[m++] = ubuf(static_cast<int64_t>(d_improper_atom4[idx])).d;
	  }

	  // Special 数据：nspecial[3] + special列表
	  int nspecial_idx = i * 3;
	  mybuf[m++] =
	      ubuf(static_cast<int64_t>(d_nspecial[nspecial_idx + 0])).d;
	  mybuf[m++] =
	      ubuf(static_cast<int64_t>(d_nspecial[nspecial_idx + 1])).d;
	  mybuf[m++] =
	      ubuf(static_cast<int64_t>(d_nspecial[nspecial_idx + 2])).d;
  
  // special 列表：读取 nspecial[2]（累积计数，即 1-2+1-3+1-4 总数）
  const int nspecial_total_raw = d_nspecial[nspecial_idx + 2];
#ifndef NDEBUG
  assert(nspecial_total_raw >= 0);
  assert(nspecial_total_raw <= maxspecial);
#endif
  int nspecial_total = nspecial_total_raw;
  if (nspecial_total < 0) nspecial_total = 0;
  if (nspecial_total > maxspecial) nspecial_total = maxspecial;
  for (int k = 0; k < nspecial_total; k++) {
    int idx = i * maxspecial + k;
    mybuf[m++] = ubuf(static_cast<int64_t>(d_special[idx])).d;
  }
}

/**
 * @brief GPU Kernel: 从 exchange buffer 解包接收的原子
 *
 * 对齐 LAMMPS AtomVecFullKokkos_UnpackExchangeFunctor::operator()
 * 每个线程处理一个接收的原子
 * 使用 atomicAdd 分配新的本地索引
 */
__global__ void UnpackExchangeTopologyKernel(
    int nrecv, const double* buf, int size_exchange, int* nlocal_ptr, int nmax,
    // 基础属性
    rbmd::Real* d_px, rbmd::Real* d_py, rbmd::Real* d_pz, rbmd::Real* d_vx,
    rbmd::Real* d_vy, rbmd::Real* d_vz, rbmd::Id* d_id, rbmd::Id* d_type,
    rbmd::Id* d_molecule, rbmd::Real* d_charge,
    // Bond 数据
    int* d_num_bond, int* d_bond_type, rbmd::Id* d_bond_atom, int bond_per_atom,
    // Angle 数据
    int* d_num_angle, int* d_angle_type, rbmd::Id* d_angle_atom1,
    rbmd::Id* d_angle_atom2, rbmd::Id* d_angle_atom3, int angle_per_atom,
    // Dihedral 数据
    int* d_num_dihedral, int* d_dihedral_type, rbmd::Id* d_dihedral_atom1,
    rbmd::Id* d_dihedral_atom2, rbmd::Id* d_dihedral_atom3,
    rbmd::Id* d_dihedral_atom4, int dihedral_per_atom,
    // Improper 数据
    int* d_num_improper, int* d_improper_type, rbmd::Id* d_improper_atom1,
    rbmd::Id* d_improper_atom2, rbmd::Id* d_improper_atom3,
    rbmd::Id* d_improper_atom4, int improper_per_atom,
    // Special 数据
    int* d_nspecial, rbmd::Id* d_special, int maxspecial) {
  
  unsigned int myrecv = blockIdx.x * blockDim.x + threadIdx.x;
  if (myrecv >= nrecv) return;

  // 分配新的本地索引（atomic 操作确保线程安全）
  int i = atomicAdd(nlocal_ptr, 1);
  if (i < 0 || i >= nmax) {
    return;
  }

  // 计算 buffer 起始位置
  const double* mybuf = buf + myrecv * size_exchange;
  int m = 0;

  // [0] 跳过 size_exchange 标记
  m++;

  // [1-3] 坐标
  d_px[i] = mybuf[m++];
  d_py[i] = mybuf[m++];
  d_pz[i] = mybuf[m++];

  // [4-6] 速度
  d_vx[i] = mybuf[m++];
  d_vy[i] = mybuf[m++];
  d_vz[i] = mybuf[m++];

  // [7-10] id, type, molecule, charge
  d_id[i] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  d_type[i] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  d_molecule[i] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  d_charge[i] = mybuf[m++];

  // Bond 数据
  int num_bond_i = ClampCountToCapacity(
      static_cast<int>(ubuf(mybuf[m++]).i), bond_per_atom);
  d_num_bond[i] = num_bond_i;
  for (int k = 0; k < num_bond_i; k++) {
    int idx = i * bond_per_atom + k;
    d_bond_type[idx] = static_cast<int>(ubuf(mybuf[m++]).i);
    d_bond_atom[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  }

  // Angle 数据
  int num_angle_i = ClampCountToCapacity(
      static_cast<int>(ubuf(mybuf[m++]).i), angle_per_atom);
  d_num_angle[i] = num_angle_i;
  for (int k = 0; k < num_angle_i; k++) {
    int idx = i * angle_per_atom + k;
    d_angle_type[idx] = static_cast<int>(ubuf(mybuf[m++]).i);
    d_angle_atom1[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_angle_atom2[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_angle_atom3[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  }

  // Dihedral 数据
  int num_dihedral_i = ClampCountToCapacity(
      static_cast<int>(ubuf(mybuf[m++]).i), dihedral_per_atom);
  d_num_dihedral[i] = num_dihedral_i;
  for (int k = 0; k < num_dihedral_i; k++) {
    int idx = i * dihedral_per_atom + k;
    d_dihedral_type[idx] = static_cast<int>(ubuf(mybuf[m++]).i);
    d_dihedral_atom1[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_dihedral_atom2[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_dihedral_atom3[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_dihedral_atom4[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  }

  // Improper 数据
  int num_improper_i = ClampCountToCapacity(
      static_cast<int>(ubuf(mybuf[m++]).i), improper_per_atom);
  d_num_improper[i] = num_improper_i;
  for (int k = 0; k < num_improper_i; k++) {
    int idx = i * improper_per_atom + k;
    d_improper_type[idx] = static_cast<int>(ubuf(mybuf[m++]).i);
    d_improper_atom1[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_improper_atom2[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_improper_atom3[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
    d_improper_atom4[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  }

  // Special 数据
  int nspecial_idx = i * 3;
  int n12 = static_cast<int>(ubuf(mybuf[m++]).i);
  int n13 = static_cast<int>(ubuf(mybuf[m++]).i);
  int n14 = static_cast<int>(ubuf(mybuf[m++]).i);
#ifndef NDEBUG
  assert(n12 >= 0 && n13 >= 0 && n14 >= 0);
  assert(n12 <= n13 && n13 <= n14);
  assert(n14 <= maxspecial);
#endif
  // clamp to safe monotonic range
  if (n12 < 0) n12 = 0;
  if (n13 < 0) n13 = 0;
  if (n14 < 0) n14 = 0;
  if (n12 > n13) n13 = n12;
  if (n13 > n14) n14 = n13;
  if (n14 > maxspecial) n14 = maxspecial;
  if (n13 > n14) n13 = n14;
  if (n12 > n13) n12 = n13;

  d_nspecial[nspecial_idx + 0] = n12;
  d_nspecial[nspecial_idx + 1] = n13;
  d_nspecial[nspecial_idx + 2] = n14;
  
  int nspecial_total = n14;
  for (int k = 0; k < nspecial_total; k++) {
    int idx = i * maxspecial + k;
    d_special[idx] = static_cast<rbmd::Id>(ubuf(mybuf[m++]).i);
  }
}

// ==================== Op 实现 ====================

void op::PackExchangeTopologyOp<device::DEVICE_GPU>::operator()(
    const TopologyPackParams& params) {
  if (params.nsend == 0) return;

  constexpr int kBlockSize = 256;
  unsigned int blocks_per_grid = (params.nsend + kBlockSize - 1) / kBlockSize;

  PackExchangeTopologyKernel<<<blocks_per_grid, kBlockSize>>>(
      params.nsend, params.sendlist, params.buf, params.size_exchange,
      params.nmax,
      // 基础属性
      params.d_px, params.d_py, params.d_pz, params.d_vx, params.d_vy,
      params.d_vz, params.d_id, params.d_type, params.d_molecule,
      params.d_charge,
      // Bond
      params.d_num_bond, params.d_bond_type, params.d_bond_atom,
      params.bond_per_atom,
      // Angle
      params.d_num_angle, params.d_angle_type, params.d_angle_atom1,
      params.d_angle_atom2, params.d_angle_atom3, params.angle_per_atom,
      // Dihedral
      params.d_num_dihedral, params.d_dihedral_type, params.d_dihedral_atom1,
      params.d_dihedral_atom2, params.d_dihedral_atom3,
      params.d_dihedral_atom4, params.dihedral_per_atom,
      // Improper
      params.d_num_improper, params.d_improper_type, params.d_improper_atom1,
      params.d_improper_atom2, params.d_improper_atom3,
      params.d_improper_atom4, params.improper_per_atom,
      // Special
      params.d_nspecial, params.d_special, params.maxspecial,
      // Position shift + Box clamping
      params.d_shift_x, params.d_shift_y, params.d_shift_z,
      params.box_min[0], params.box_min[1], params.box_min[2],
      params.box_max[0], params.box_max[1], params.box_max[2]);

  // 同步以确保 kernel 完成（可选，根据需要）
  CHECK_RUNTIME(DEVICESYNC());
}

void op::UnpackExchangeTopologyOp<device::DEVICE_GPU>::operator()(
    const TopologyUnpackParams& params) {
  if (params.nrecv == 0) return;

  constexpr int kBlockSize = 256;
  unsigned int blocks_per_grid = (params.nrecv + kBlockSize - 1) / kBlockSize;

  UnpackExchangeTopologyKernel<<<blocks_per_grid, kBlockSize>>>(
      params.nrecv, params.buf, params.size_exchange, params.nlocal_ptr,
      params.nmax,
      // 基础属性
      params.d_px, params.d_py, params.d_pz, params.d_vx, params.d_vy,
      params.d_vz, params.d_id, params.d_type, params.d_molecule,
      params.d_charge,
      // Bond
      params.d_num_bond, params.d_bond_type, params.d_bond_atom,
      params.bond_per_atom,
      // Angle
      params.d_num_angle, params.d_angle_type, params.d_angle_atom1,
      params.d_angle_atom2, params.d_angle_atom3, params.angle_per_atom,
      // Dihedral
      params.d_num_dihedral, params.d_dihedral_type, params.d_dihedral_atom1,
      params.d_dihedral_atom2, params.d_dihedral_atom3,
      params.d_dihedral_atom4, params.dihedral_per_atom,
      // Improper
      params.d_num_improper, params.d_improper_type, params.d_improper_atom1,
      params.d_improper_atom2, params.d_improper_atom3,
      params.d_improper_atom4, params.improper_per_atom,
      // Special
      params.d_nspecial, params.d_special, params.maxspecial);

  // 避免重复同步（性能成本高）：
  // 如需在 host 侧读取 unpack 结果，应由上层在 D2H/MPI 边界处统一同步。
}
