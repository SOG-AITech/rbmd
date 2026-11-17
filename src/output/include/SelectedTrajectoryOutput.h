#pragma once
#include <fstream>
#include "output.h"
#include "trajectory_selection.h"

// Synchronous bounded device compaction. MPI is only used by the simulation
// thread; no whole-system trajectory or per-native-atom selection mask exists.
class SelectedTrajectoryOutput : public Output {
 public:
  void Init() override;
  void Execute(rbmd::Id timestep) override;

 private:
  rbmd::trajectory::Predicate MakePredicate();
  void FreezeIds(const std::vector<rbmd::Id>& local_ids);
  void CollectiveCheck(bool ok, const std::string& message) const;
  rbmd::GlobalCount GlobalSum(rbmd::GlobalCount local) const;

  rbmd::trajectory::Options _options;
  rbmd::Id _interval{1};
  int _rank{0}, _world{1};
  bool _frozen{false};
  std::ofstream _file;
  thrust::device_vector<rbmd::Id> _ids, _types;
  thrust::device_vector<rbmd::trajectory::Region> _regions;
  thrust::device_vector<size_t> _indices;
  thrust::device_vector<rbmd::Id> _id_scratch;
  thrust::device_vector<rbmd::Real> _real_scratch;
};
