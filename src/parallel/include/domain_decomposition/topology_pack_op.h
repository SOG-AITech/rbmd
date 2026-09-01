#pragma once

#include "common/device_types.h"
#include "common/types.h"

namespace op {

/**
 * @brief 计算单个原子在 exchange buffer 中的大小（单位：double）
 *
 * 对齐 LAMMPS size_exchange，包含：
 * - buffer 长度标记：1（写入 mybuf[0]）
 * - 基础属性：x(3) + v(3) + id + type + molecule + charge = 10
 * - num_bond(1) + 每个bond占2个(type+atom) = 1 + 2*bond_per_atom
 * - num_angle(1) + 每个angle占4个(type+atom1/2/3) = 1 + 4*angle_per_atom
 * - num_dihedral(1) + 每个dihedral占5个(type+atom1/2/3/4) = 1 + 5*dihedral_per_atom
 * - num_improper(1) + 每个improper占5个(type+atom1/2/3/4) = 1 + 5*improper_per_atom
 * - nspecial(3) + special列表 = 3 + maxspecial
 */
inline int ComputeSizeExchange(int bond_per_atom, int angle_per_atom,
                                int dihedral_per_atom, int improper_per_atom,
                                int maxspecial) {
  // 固定项（单位：double）
  // 1(size_exchange) + 10(base attrs) + 4(num_bond/angle/dihedral/improper) + 3(nspecial)
  constexpr int kBase = 18;
  return kBase + maxspecial + 2 * bond_per_atom + 4 * angle_per_atom +
         5 * dihedral_per_atom + 5 * improper_per_atom;
}

/**
 * @brief 结构体：打包拓扑数据到 exchange buffer 的参数
 * 
 * 将众多参数封装到结构体中，避免函数参数过多
 */
struct TopologyPackParams {
  int nsend;              // 要发送的原子数量
  const int* sendlist;    // 要发送的原子索引列表 (device)
  double* buf;            // 打包缓冲区 (device, 扁平化 [nsend * size_exchange])
  int size_exchange;      // 每个原子的 buffer 大小
  int nmax;               // per-atom 拓扑数组容量（防越界）

  // 基础属性 (device)
  const rbmd::Real* d_px;
  const rbmd::Real* d_py;
  const rbmd::Real* d_pz;
  const rbmd::Real* d_vx;
  const rbmd::Real* d_vy;
  const rbmd::Real* d_vz;
  const rbmd::Id* d_id;
  const rbmd::Id* d_type;
  const rbmd::Id* d_molecule;
  const rbmd::Real* d_charge;

  // Bond 数据 (device, 2D 扁平化)
  const int* d_num_bond;
  const int* d_bond_type;
  const rbmd::Id* d_bond_atom;
  int bond_per_atom;

  // Angle 数据 (device, 2D 扁平化)
  const int* d_num_angle;
  const int* d_angle_type;
  const rbmd::Id* d_angle_atom1;
  const rbmd::Id* d_angle_atom2;
  const rbmd::Id* d_angle_atom3;
  int angle_per_atom;

  // Dihedral 数据 (device, 2D 扁平化)
  const int* d_num_dihedral;
  const int* d_dihedral_type;
  const rbmd::Id* d_dihedral_atom1;
  const rbmd::Id* d_dihedral_atom2;
  const rbmd::Id* d_dihedral_atom3;
  const rbmd::Id* d_dihedral_atom4;
  int dihedral_per_atom;

  // Improper 数据 (device, 2D 扁平化)
  const int* d_num_improper;
  const int* d_improper_type;
  const rbmd::Id* d_improper_atom1;
  const rbmd::Id* d_improper_atom2;
  const rbmd::Id* d_improper_atom3;
  const rbmd::Id* d_improper_atom4;
  int improper_per_atom;

  // Special 数据 (device, 2D 扁平化)
  const int* d_nspecial;
  const rbmd::Id* d_special;
  int maxspecial;

  // ---- Position shift（在 pack 时内部应用，避免污染源数据） ----
  // 每个发送原子的位移量 (device, 大小 nsend)，索引对应 sendlist 中的位置
  // 若为 nullptr 则不做 shift（向后兼容）
  const rbmd::Real* d_shift_x = nullptr;
  const rbmd::Real* d_shift_y = nullptr;
  const rbmd::Real* d_shift_z = nullptr;
  // 全局盒子边界，用于 shift 后的边界钳位 (clamping)
  rbmd::Real box_min[3] = {0, 0, 0};
  rbmd::Real box_max[3] = {0, 0, 0};
};

/**
 * @brief 结构体：解包拓扑数据从 exchange buffer 的参数
 */
struct TopologyUnpackParams {
  int nrecv;           // 接收的原子数量
  const double* buf;   // 接收缓冲区 (device, 扁平化 [nrecv * size_exchange])
  int size_exchange;   // 每个原子的 buffer 大小
  int* nlocal_ptr;     // 当前本地原子数 (device, 单值，用于 atomic 分配)
  int nmax;            // per-atom 拓扑数组容量（防越界）

  // 基础属性 (device)
  rbmd::Real* d_px;
  rbmd::Real* d_py;
  rbmd::Real* d_pz;
  rbmd::Real* d_vx;
  rbmd::Real* d_vy;
  rbmd::Real* d_vz;
  rbmd::Id* d_id;
  rbmd::Id* d_type;
  rbmd::Id* d_molecule;
  rbmd::Real* d_charge;

  // Bond 数据 (device, 2D 扁平化)
  int* d_num_bond;
  int* d_bond_type;
  rbmd::Id* d_bond_atom;
  int bond_per_atom;

  // Angle 数据 (device, 2D 扁平化)
  int* d_num_angle;
  int* d_angle_type;
  rbmd::Id* d_angle_atom1;
  rbmd::Id* d_angle_atom2;
  rbmd::Id* d_angle_atom3;
  int angle_per_atom;

  // Dihedral 数据 (device, 2D 扁平化)
  int* d_num_dihedral;
  int* d_dihedral_type;
  rbmd::Id* d_dihedral_atom1;
  rbmd::Id* d_dihedral_atom2;
  rbmd::Id* d_dihedral_atom3;
  rbmd::Id* d_dihedral_atom4;
  int dihedral_per_atom;

  // Improper 数据 (device, 2D 扁平化)
  int* d_num_improper;
  int* d_improper_type;
  rbmd::Id* d_improper_atom1;
  rbmd::Id* d_improper_atom2;
  rbmd::Id* d_improper_atom3;
  rbmd::Id* d_improper_atom4;
  int improper_per_atom;

  // Special 数据 (device, 2D 扁平化)
  int* d_nspecial;
  rbmd::Id* d_special;
  int maxspecial;
};

/**
 * @brief Op: 将要迁移的原子打包到 exchange buffer
 *
 * 对齐 LAMMPS AtomVecFullKokkos::pack_exchange，在 GPU 上直接打包拓扑数据。
 * 使用 ubuf trick 将整数编码进 double buffer。
 *
 * Buffer 布局（对齐 LAMMPS）：
 * [0] size_exchange (buffer 长度标记)
 * [1-3] x, y, z
 * [4-6] vx, vy, vz
 * [7] id (encoded)
 * [8] type (encoded)
 * [9] molecule (encoded)
 * [10] charge
 * [11] num_bond (encoded)
 * [12...] bond_type[k], bond_atom[k] (循环，encoded)
 * [...] num_angle (encoded)
 * [...] angle_type[k], angle_atom1/2/3[k] (循环，encoded)
 * [...] num_dihedral (encoded)
 * [...] dihedral_type[k], dihedral_atom1/2/3/4[k] (循环，encoded)
 * [...] num_improper (encoded)
 * [...] improper_type[k], improper_atom1/2/3/4[k] (循环，encoded)
 * [...] nspecial[0], nspecial[1], nspecial[2] (encoded)
 * [...] special[k] (循环，encoded)
 */
template <typename DEVICE>
struct PackExchangeTopologyOp {
  void operator()(const TopologyPackParams& params);
};

/**
 * @brief Op: 从 exchange buffer 解包接收的原子
 *
 * 对齐 LAMMPS AtomVecFullKokkos::unpack_exchange，在 GPU 上直接解包拓扑数据。
 * 使用 atomic 操作分配新原子的本地索引。
 *
 * 解包顺序与 pack 一致。
 * 解包后 nlocal_ptr 指向更新后的本地原子数。
 */
template <typename DEVICE>
struct UnpackExchangeTopologyOp {
  void operator()(const TopologyUnpackParams& params);
};

// ==================== GPU Specialization ====================
#pragma region GPU
template <>
struct PackExchangeTopologyOp<device::DEVICE_GPU> {
  void operator()(const TopologyPackParams& params);
};

template <>
struct UnpackExchangeTopologyOp<device::DEVICE_GPU> {
  void operator()(const TopologyUnpackParams& params);
};
#pragma endregion

}  // namespace op
