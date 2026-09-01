#pragma once
#include <thrust/device_vector.h>

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "collective_communicator/collective_communicator.h"
#include "domain_decomposition/domain_decomposition.h"
#include "domain_decomposition/forward_ghost_state.h"
#include "global_structure_info.h"

class RbmdParallelUntil {
 private:
  std::shared_ptr<LinkedCell> _linked_cell;
  std::random_device rd;
  std::mt19937 _rbe_seed_generator;
  std::uint64_t _rbe_seed_sequence{0};
  unsigned int _rbe_base_seed{0};
  bool _rbe_seed_ready{false};
  int _rbe_p_number{};

 public:
  RbmdParallelUntil() = default;
  ~RbmdParallelUntil() = default;
  std::shared_ptr<DomainDecomposition> _domdec;

  void Init(CollectiveCommunicator::Backend backend);
  void SetUp();

  std::shared_ptr<CollectiveCommunicator> _communicator;

  int FindAtomOwnerRank(rbmd::Real px, rbmd::Real py, rbmd::Real pz) const;

  unsigned int GetDeterministicRandomSeed();

  void GetRbeRadomM(rbmd::Real alpha, bool use_random, const Box& sample_box,
                    const std::array<rbmd::Id, 3>& sample_multipliers,
                    std::vector<rbmd::Real>& x, std::vector<rbmd::Real>& y,
                    std::vector<rbmd::Real>& z);

  void BalanceAndExchange();

  // Build the fixed owner/ghost schedule after a full halo rebuild and atom
  // reorder. The schedule remains valid until the next ownership exchange.
  bool PrepareForwardCoordinateExchange();

  // Refresh coordinates in the existing ghost slots without changing atom
  // ownership, ordering, or the halo topology.
  void ForwardExchangeCoordinates();

  // Refresh one fused SHAKE state bundle in the existing ghost slots.
  void ForwardExchangeGhostState(ForwardGhostState state);

  // Epoch of the fixed owner/ghost schedule. It changes only after a full
  // ownership/halo exchange and lets consumers cache local indices safely.
  std::uint32_t ForwardExchangeEpoch() const;

  // Propagate an owner-computed per-atom scalar to the current ghost copies.
  void ForwardExchangeAtomScalar(thrust::device_vector<rbmd::Real>& values);

  GlobalStructureInfo _global_structure_info;

 private:
  unsigned int GetRbeGlobalRandomSeed();
};
