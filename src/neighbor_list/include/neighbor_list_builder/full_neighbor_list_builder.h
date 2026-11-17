#pragma once
#include <array>
#include <thrust/device_vector.h>

#include "../neighbor_list/neighbor_list.h"
#include "base_neighbor_list_builder.h"

class FullNeighborListBuilder : public BaseNeighborListBuilder {
 public:
  explicit FullNeighborListBuilder();

  std::shared_ptr<NeighborList> Build() override;
  std::shared_ptr<NeighborList> Build(rbmd::Real custom_cutoff) override;

 protected:
  void ComputeNeighborCells() override;

  void ComputeNeighborCellsWithoutPBC() override;

  void EstimateNeighborsList() override;

  rbmd::Id GenerateNeighborsList() override;

  rbmd::Id ActiveAtomCount() const;
  rbmd::Id InitialNeighborCapacity(rbmd::Real cutoff) const;
  rbmd::Id GrownNeighborCapacity(rbmd::Id required) const;
  rbmd::Id ChooseThreadsPerParticle() const;
  rbmd::Id ChooseThreadsPerParticle(rbmd::Id estimated_work_items,
                                    const char* override_env,
                                    rbmd::Id minimum_tpp) const;
  void PrepareFixedWidthList(const std::shared_ptr<NeighborList>& list,
                             rbmd::Id active_atoms, rbmd::Id capacity);
  void EnsureNeighborCellsCurrent();

  std::shared_ptr<DeviceData> _device_data;
  rbmd::Id _required_neighbor_capacity = 0;
  thrust::device_vector<rbmd::Id> _d_required_neighbor_capacity{};

 private:
  std::array<rbmd::Id, 14> NeighborCellGridSignature() const;
  std::array<rbmd::Real, 4> NeighborStencilGeometrySignature() const;
  thrust::device_vector<rbmd::Id> _d_neighbor_stencil_indices{};
  std::array<rbmd::Id, 14> _neighbor_cell_grid_signature{};
  std::array<rbmd::Real, 4> _neighbor_stencil_geometry_signature{};
  bool _neighbor_cells_valid = false;
};
