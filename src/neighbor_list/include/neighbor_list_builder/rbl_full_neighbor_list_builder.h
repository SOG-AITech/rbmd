#pragma once
#include <functional>

#include "data_manager.h"
#include "full_neighbor_list_builder.h"

class RblFullNeighborListBuilder : public FullNeighborListBuilder {
 public:
  explicit RblFullNeighborListBuilder();
  std::shared_ptr<NeighborList> Build() override;
  // Invokes on_stable_atoms exactly once after migration/halo updates and the
  // atom-id map are complete. The remaining build only reads simulation atom
  // arrays, so the callback may enqueue RBE and bonded work on other streams.
  // On reuse, on_native_atoms runs before coordinate forwarding. It may only
  // read native particle data; ghost coordinates are not current until the
  // on_stable_atoms callback. No atom storage resize/reorder occurs in between.
  std::shared_ptr<NeighborList> BuildWithStableAtoms(
      const std::function<void()>& on_stable_atoms,
      const std::function<void()>& on_native_atoms = {});
  std::shared_ptr<NeighborList> BuildFromCurrentGhosts();
  std::shared_ptr<NeighborList> Build(rbmd::Real custom_cutoff) override;
 protected:
  virtual bool SupportsCandidateCache() const { return true; }
  std::shared_ptr<NeighborList> BuildImpl(
      bool exchange_ghosts,
      const std::function<void()>& on_stable_atoms,
      const std::function<void()>& on_native_atoms = {});
  rbmd::Real _r_core = 0;
  rbmd::Id _neighbor_sample_num = 0;
  rbmd::Real _system_rho = 0;
  rbmd::Id _selection_frequency = 0;
  rbmd::Id _global_atoms_num = 0;
  rbmd::Id _random_neighbor_capacity = 0;
  rbmd::Id _required_candidate_capacity = 0;
  rbmd::Id _candidate_cache_max_per_atom = 0;
  bool _compact_cache_seeded = false;
  bool _candidate_cache_fixed_width = false;
  bool _candidate_cache_implicit_offsets = false;
  bool _fixed_core_generated_this_build = false;
  thrust::device_vector<rbmd::Id> _d_required_random_neighbor_capacity;
  thrust::device_vector<rbmd::Id> _d_fixed_candidate_status;
  std::shared_ptr<NeighborList> _candidate_list;

  void EstimateNeighborsList() override;
  rbmd::Id GenerateNeighborsList() override;
  void EstimateCandidateList();
  rbmd::Id GenerateCandidateList();
  void PrepareCandidateFixedStorage(rbmd::Id active_atoms,
                                    rbmd::Id capacity);
  rbmd::Id GenerateGuardedFixedCandidateList();
  rbmd::Id GenerateGuardedFixedCandidateAndCoreList();
  void GenerateCompactCandidateList();
  bool TryGenerateAppendedCompactCandidateList();
  rbmd::Id FilterCandidateList();
  void SampleCandidateList();
  void GetRblParams();


};
