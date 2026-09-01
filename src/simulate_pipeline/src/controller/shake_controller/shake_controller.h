#pragma once
#include <cstdint>
#include <limits>

#include "types.h"
#include "vector"
#include "data_manager.h"
#include "model/md_data.h"

class ShakeController {
 public:
enum class ShakeCommStage {
    PredictedPositionForward,
    VelocityForward,
    VelocityCorrectedForward,
    CorrectionReverse,
    CorrectedStateForward,
};

  ShakeController();
  virtual ~ShakeController() = default;

  void Init();
  void ShakeA();
  void ShakeB();

protected:
  std::shared_ptr<StructureInfoData> _structure_info_data;
  std::shared_ptr<DeviceData> _device_data;

  rbmd::Real _dt;
  rbmd::Real _fmt2v;

  std::vector<rbmd::Real> _shake_px;
  std::vector<rbmd::Real> _shake_py;
  std::vector<rbmd::Real> _shake_pz;

  std::vector<rbmd::Real> _shake_vx;
  std::vector<rbmd::Real> _shake_vy;
  std::vector<rbmd::Real> _shake_vz;

  std::shared_ptr<Box> _box;

  bool UseMpiOwnedShakePath() const;
  void BuildOwnedShakeClustersForMpi();
  void ResolveOwnedShakeClusterIndicesForMpi(ShakeCommStage stage);
  void ForwardSyncShakeState(ShakeCommStage stage);
  void ForwardSyncReplicatedShakeState(ShakeCommStage stage);
  void ReverseAccumulateShakeCorrections(ShakeCommStage stage);
  void EnsureReplicatedShakeClusters();
  void ZeroShakeCorrections();
  void RunLegacySerialShakeA();
  void RunLegacySerialShakeB();
  void RunReplicatedShakeAKernel();
  void RunReplicatedShakeBKernel();
  void SnapshotReplicatedShakeKernelIndices();
  void ApplyReplicatedShakeCorrections(bool apply_position);
  void EmitShakeResidualDebugCsv(const char* stage, ShakeCommStage resolve_stage);

  std::uint32_t _shake_cluster_epoch{
      std::numeric_limits<std::uint32_t>::max()};
  rbmd::Id _shake_cluster_native_atoms{-1};
  rbmd::Id _shake_cluster_total_atoms{-1};
  int _shake_cluster_angle_per_atom{-1};
};
