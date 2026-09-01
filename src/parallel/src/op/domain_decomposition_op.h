#pragma once
#include "common/device_types.h"
#include "common/rbmd_define.h"
#include "communication_partner.h"
#include "data_manager/include/model/box.h"

namespace op {
template <typename DEVICE>
struct HandleDomainLeavingAtomsOneDimOp {
  void operator()(rbmd::Real* d_start_region, rbmd::Real* d_end_region,
                  rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  rbmd::Real* pos_to_modify, Box d_box, rbmd::Id dim,
                  rbmd::Real shift, rbmd::Id total_atoms_num);
};

template <typename DEVICE>
struct PopulateHaloOp {
  void operator()(rbmd::Real* d_start_region,
                  rbmd::Real* d_end_region, rbmd::Real* px,
                  rbmd::Real* py, rbmd::Real* pz, Box d_box,
                  rbmd::Real* halo_px, rbmd::Real* halo_py,
                  rbmd::Real* halo_pz, unsigned int* counter,
                  unsigned max_halo_size, unsigned dim, rbmd::Real shift,
                  rbmd::Id native_atoms_num,
                  // 拓扑相关参数：复制 atom_id 和 atom_type 用于 bond/angle 力计算
                  rbmd::Id* atoms_id = nullptr,
                  rbmd::Id* atoms_type = nullptr,
                  rbmd::Id* halo_atoms_id = nullptr,
                  rbmd::Id* halo_atoms_type = nullptr);
};

template <typename DEVICE>
struct MarkLeavingAtomsInRegionOp {
  void operator()(rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  PositionInfo* d_position_infos,
                  rbmd::Id mark_atoms_num, int* leaving_flags);
};





template <typename DEVICE>
struct ProcessLeavingAtomOp {
  void operator()(rbmd::Real* p_px, rbmd::Real* p_py,
                  rbmd::Real* p_pz, PositionInfo* d_position_infos,
                   rbmd::Id total_leaving, Box d_domain);
};

template <typename DEVICE>
struct MarkHaloAtomsInRegionOp {
  void operator()(rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  PositionInfo* d_position_infos,
                  rbmd::Id mark_atoms_num, int* halo_flags,
                  int* halo_owner, int info_index);
};

template <typename DEVICE>
struct ProcessHaloAtomOp {
  void operator()(rbmd::Real* p_px, rbmd::Real* p_py,
                  rbmd::Real* p_pz, PositionInfo* d_position_infos,
                  rbmd::Id total_halo, Box d_domain,
                  const int* halo_owner, int owner_idx);
};



template <typename DEVICE>
struct CountHaloCandidatesOp {
  void operator()(const rbmd::Real* d_start_region, // 设备端内存，定义搜索区域的起始边界
                  const rbmd::Real* d_end_region, // 设备端内存，定义搜索区域的结束边界
                  const rbmd::Real* px, // 所有本地原子的x坐标数组 (设备端)
                  const rbmd::Real* py, // 所有本地原子的y坐标数组 (设备端)
                  const rbmd::Real* pz, // 所有本地原子的z坐标数组 (设备端)
                  rbmd::Id native_atoms_num, // 本地域中原子的总数
                  unsigned int* d_global_count);
};

#pragma region GPU
template <>
struct HandleDomainLeavingAtomsOneDimOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real* d_start_region, rbmd::Real* d_end_region,
                  rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  rbmd::Real* pos_to_modify, Box d_box, rbmd::Id dim,
                  rbmd::Real shift, rbmd::Id total_atoms_num);
};

template <>
struct PopulateHaloOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real* d_start_region,
                  rbmd::Real* d_end_region, rbmd::Real* px,
                  rbmd::Real* py, rbmd::Real* pz, Box d_box,
                  rbmd::Real* halo_px, rbmd::Real* halo_py,
                  rbmd::Real* halo_pz, unsigned int* counter,
                  unsigned max_halo_size, unsigned dim, rbmd::Real shift,
                  rbmd::Id native_atoms_num,
                  // 拓扑相关参数：复制 atom_id 和 atom_type 用于 bond/angle 力计算
                  rbmd::Id* atoms_id = nullptr,
                  rbmd::Id* atoms_type = nullptr,
                  rbmd::Id* halo_atoms_id = nullptr,
                  rbmd::Id* halo_atoms_type = nullptr);
};

template <>
struct MarkLeavingAtomsInRegionOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  PositionInfo* d_position_infos,
                  rbmd::Id native_atoms_num, int* leaving_flags);
};



template <>
struct ProcessLeavingAtomOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real* p_px, rbmd::Real* p_py,
                  rbmd::Real* p_pz, PositionInfo* d_position_infos,
                   rbmd::Id total_leaving, Box d_domain);
};

template <>
struct MarkHaloAtomsInRegionOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real* px, rbmd::Real* py, rbmd::Real* pz,
                  PositionInfo* d_position_infos,
                  rbmd::Id mark_atoms_num, int* halo_flags,
                  int* halo_owner, int info_index);
};

template <>
struct ProcessHaloAtomOp<device::DEVICE_GPU> {
  void operator()(rbmd::Real* p_px, rbmd::Real* p_py,
                  rbmd::Real* p_pz, PositionInfo* d_position_infos,
                  rbmd::Id total_halo, Box d_domain,
                  const int* halo_owner, int owner_idx);
};




template <>
struct CountHaloCandidatesOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Real* d_start_region, // 设备端内存，定义搜索区域的起始边界
                  const rbmd::Real* d_end_region, // 设备端内存，定义搜索区域的结束边界
                  const rbmd::Real* px, // 所有本地原子的x坐标数组 (设备端)
                  const rbmd::Real* py, // 所有本地原子的y坐标数组 (设备端)
                  const rbmd::Real* pz, // 所有本地原子的z坐标数组 (设备端)
                  rbmd::Id native_atoms_num, // 本地域中原子的总数
                  unsigned int* d_global_count);
};


#pragma endregion
} // namespace op
