#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../../data_manager/include/model/box.h"
#include "cell.h"
#include "common/rbmd_define.h"
#include "common/types.h"
#include "config_data.h"
#include "model/device_data.h"
#include "model/structure_info_data.h"

struct LinkedCellDeviceDataPtr {
  rbmd::Id _d_total_cells = 0;
  rbmd::Id ALIGN(ALIGN_SIZE(rbmd::Id, 3)) _d_per_dimension_cells[3]{};
  rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) _d_cell_length[3]{};
  rbmd::Real _d_cutoff = 0;
  // Cell* _d_cells = nullptr;
  rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) _d_cell_length_reciprocal[3]{};
  rbmd::Id ALIGN(ALIGN_SIZE(rbmd::Id, 3)) _d_nativate_cells_dims[3]{};
  rbmd::Id ALIGN(ALIGN_SIZE(rbmd::Id, 3)) _d_nativate_cells_start[3]{};
  bool ALIGN(ALIGN_SIZE(bool, 3)) _d_covers_whole_domain[3]{};
};

class LinkedCell {
 public:
  //  LinkedCell(const LinkedCell&) = delete;
  //  LinkedCell& operator=(const LinkedCell&) = delete;
  //  static LinkedCell& GetInstance()
  //  {
  //    static LinkedCell instance;
  //    return instance;
  //  }
  // private:
  LinkedCell();
  ~LinkedCell();

  /// 总共的cell数量（包括HaloCell）
  rbmd::Id _total_cells = 0;
  rbmd::Id _total_atoms_num = 0;
  rbmd::Id _native_atoms_num = 0;
  rbmd::Id _ghost_atoms_num = 0;
  /// 各维度的cell的数量（包括HaloCell）
  rbmd::Id ALIGN(ALIGN_SIZE(rbmd::Id, 3)) _per_dimension_cells[3]{};
  /// 各维度cell的长度
  rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) _cell_length[3]{};
  /// 各维度cell的长度的倒数
  rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) _cell_length_reciprocal[3]{};
  rbmd::Real _halo_cutoff = 0;
  rbmd::Real ALIGN(ALIGN_SIZE(rbmd::Real, 3)) _halo_length[3]{};
  rbmd::Id ALIGN(ALIGN_SIZE(rbmd::Id, 3)) _nativate_cells[3]{};
  rbmd::Id ALIGN(ALIGN_SIZE(rbmd::Id, 3)) _nativate_cell_start[3]{};
  rbmd::Id ALIGN(ALIGN_SIZE(rbmd::Id, 3)) _nativate_cell_end[3]{};
  bool ALIGN(ALIGN_SIZE(bool, 3)) _covers_whole_domain[3]{false, false, false};

  // 每个cell内的原子在原子列表的开始索引
  thrust::device_vector<rbmd::Id> _in_atom_list_start_index{};
  // 每个cell内的原子在原子列表的结束索引
  thrust::device_vector<rbmd::Id> _in_atom_list_end_index{};

  thrust::device_vector<Cell> _cells{};
  // 存储邻居
  thrust::device_vector<rbmd::Id> _neighbor_cell{};
  thrust::device_vector<rbmd::Id> _atom_id_to_idx{};
  rbmd::Real _cutoff = 0;
  rbmd::Real _skin = 0;
  rbmd::Real _neighbor_cutoff = 0;
  rbmd::Id _cell_count_within_cutoff = 1;

  thrust::device_vector<rbmd::Id> _per_atom_cell_id{};
  thrust::device_vector<rbmd::Id> _cell_sorted_cell_id{};
  thrust::device_vector<rbmd::Id> _cell_sorted_atom_indices{};
  __host__ void Build();

  LinkedCellDeviceDataPtr* GetDataPtr();

  void InitializeCells();
  void AssignAtomsToCell();

  void ComputeCellRangesIndices();

  void SyncHToD();

  void SortAtomsByCellKey();
  void UpdateLeavingNum(int change_leaving_num);
  void UpdateGhostNum(rbmd::Id change_ghost_num);
  void ResizeDataLeaving(size_t leaving_size);
  void EnsureCapacity(rbmd::Id required_total_atoms);
  void ResizeNativeDependentVectors(rbmd::Id target_native_atoms,
                                    rbmd::Id target_total_atoms);
  void ClearDataHalo();
  void ResizeDataHalo(size_t halo_size);
  void SetHaloCutoff(rbmd::Real halo_cutoff);
  void SetCoversWholeDomain(int dim, bool covers);

  /**
   * 由于SortAtomsByCellKey做了映射，读取文件计算键角力(在邻居力计算之后)需要将Id进行一次映射。调用这个函数。
   * @param d_target 需要映射的原子ID的数组
   * TODO 并行考虑范围问题？
   */
  template <typename T>
  void MapAtomId(thrust::device_vector<T>& d_target);

  /**
   * 由于SortAtomsByCellKey做了映射,如果计算键力角力在构建邻居列表之前(或同时)，这力加和时需要调用这个函数
   * @param d_target 键角力数组
   */
  template <typename T>
 void MapAtomIdForce(thrust::device_vector<T>& d_target);

 private:
  LinkedCellDeviceDataPtr* _linked_cell_device_data_ptr = nullptr;
  std::shared_ptr<DeviceData> _device_data;
  std::shared_ptr<StructureInfoData> _structure_info_data;
  std::shared_ptr<ConfigData> _config_data;
  void AllocDeviceMemory();
  void ResizePerAtomVectors(rbmd::Id target_total_atoms);

  std::shared_ptr<Box> _box;   // TODO 可能不太适合 待重构
};
