#pragma once
#include <thrust/device_vector.h>

#include <string>

#include "../../data_manager/include/model/box.h"
#include "../common/object.h"
#include "../common/types.h"
#include "../neighbor_list/include/linked_cell/linked_cell.h"
#include "../neighbor_list/neighbor_list.h"

// linked Cell should new in out
class BaseNeighborListBuilder : public Object {
 public:
  explicit BaseNeighborListBuilder();
  ~BaseNeighborListBuilder() override;

  virtual std::shared_ptr<NeighborList> Build() = 0;
  virtual std::shared_ptr<NeighborList> Build(rbmd::Real custom_cut_off) = 0;

 protected:
  std::shared_ptr<LinkedCell> _linked_cell;
  std::shared_ptr<NeighborList> _neighbor_list = nullptr;

  virtual void ComputeNeighborCells() = 0;

  virtual void ComputeNeighborCellsWithoutPBC() = 0;

  virtual void EstimateNeighborsList() = 0;

  virtual rbmd::Id GenerateNeighborsList() = 0;

  void ReductionSum(rbmd::Id* d_src_array, rbmd::Id* d_dst, rbmd::Id size);

  void InitNeighborListIndices();

  void ValidateCellGridSupportsPBC() const;
  bool ShouldUseStrictMPIGridValidation() const;
  bool ShouldReuseNeighborList(rbmd::Real neighbor_cutoff);
  void RecordNeighborBuildState(rbmd::Real neighbor_cutoff);
  bool HasNeighborCache() const { return _has_neighbor_cache; }
  void InvalidateNeighborCache() { _has_neighbor_cache = false; }

  rbmd::Id _neighbor_cell_num = 0;
  rbmd::Id should_realloc = RBMD_TRUE;
  std::shared_ptr<Box> _box;  // TODO 可能不太适合 待重构
  rbmd::Id* _d_should_realloc;
  rbmd::Real _trunc_distance_power_2 =
      0;  // 生成邻居的截断距离平方 通常为cutoff平方，rbl时为rcore平方

 private:
  bool CanReuseNeighborList() const;
  rbmd::Id ActiveAtomsForNeighborCache() const;
  void EmitNeighborSkinCacheDebug(const char* action,
                                  rbmd::Real neighbor_cutoff,
                                  rbmd::Id active_atoms) const;

  bool _has_neighbor_cache = false;
  rbmd::Id _cached_active_atoms = 0;
  rbmd::Id _cached_total_atoms = 0;
  rbmd::Real _cached_neighbor_cutoff = 0;
  Box _cached_force_box{};
  bool _cached_force_box_valid = false;
  rbmd::Real _last_max_displacement_sq = rbmd::Real(-1);
  rbmd::Real _last_rebuild_threshold_sq = rbmd::Real(-1);
  rbmd::Id _neighbor_cache_reuse_count = 0;
  rbmd::Id _neighbor_cache_rebuild_count = 0;
  std::string _last_cache_reason{"not_checked"};
  thrust::device_vector<rbmd::Real> _cached_px{};
  thrust::device_vector<rbmd::Real> _cached_py{};
  thrust::device_vector<rbmd::Real> _cached_pz{};
  thrust::device_vector<rbmd::Id> _cached_atom_ids{};
};
