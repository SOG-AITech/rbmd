#pragma once
#include "data_manager.h"
#include "full_neighbor_list_builder.h"

class RblFullNeighborListBuilder : public FullNeighborListBuilder {
 public:
  explicit RblFullNeighborListBuilder();
  std::shared_ptr<NeighborList> Build() override;
  std::shared_ptr<NeighborList> BuildFromCurrentGhosts();
  std::shared_ptr<NeighborList> Build(rbmd::Real custom_cutoff) override;
protected:
  virtual bool SupportsCandidateCache() const { return true; }
  std::shared_ptr<NeighborList> BuildImpl(bool exchange_ghosts);
  rbmd::Real _r_core = 0;
  rbmd::Id _neighbor_sample_num = 0;
  rbmd::Real _system_rho = 0;
  rbmd::Id _selection_frequency = 0;
  rbmd::Id _global_atoms_num = 0;
  rbmd::Id _random_neighbor_capacity = 0;
  thrust::device_vector<rbmd::Id> _d_required_random_neighbor_capacity;
  std::shared_ptr<NeighborList> _candidate_list;

  void EstimateNeighborsList() override;
  rbmd::Id GenerateNeighborsList() override;
  void EstimateCandidateList();
  rbmd::Id GenerateCandidateList();
  rbmd::Id FilterCandidateList();
  void GetRblParams();


};
