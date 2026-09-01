#pragma once
#include <array>
#include <cstdint>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "base_neighbour_communication_scheme.h"
#include "common/rbmd_define.h"
#include "data_manager.h"
#include "domain_decomposition.h"
#include "domain_decomposition/forward_ghost_state.h"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"
class IndirectNeighbourCommunicationScheme
    : public BaseNeighbourCommunicationScheme {
protected:
  double _pair_cutoff = 0.0;     // 非键相互作截断
  double _topology_cutoff = 0.0; // 拓扑相互作用阶段
  double _ghost_cutoff =
      0.0; // 实际上的ghost阶段，扩大ghost层来覆盖拓扑相互作用，
  // lammps中也是采用了这种方法，没有额外的通信bond，这个情况不是很常见

  enum class Phase : uint8_t { Leaving = 0, Halo = 1 };

  static MessageType PhaseToMessageType(Phase phase);

  struct Header {
    int count{0};
    int nbytes{0};
    uint32_t epoch{0};
  };

  struct MsgTags {
    int base{9500};
    int stride_dim{32};
    int stride_phase{8};
    int header_channel{0};
    int payload_channel{1};

    int HeaderTag(unsigned short d, Phase phase, uint32_t epoch,
                  int receive_side) const {
      return base + static_cast<int>(d) * stride_dim +
             static_cast<int>(phase) * stride_phase + 2 * header_channel +
             receive_side +
             static_cast<int>(epoch % 2) * stride_dim * stride_phase;
    }

    int PayloadTag(unsigned short d, Phase phase, uint32_t epoch,
                   int receive_side) const {
      return base + static_cast<int>(d) * stride_dim +
             static_cast<int>(phase) * stride_phase + 2 * payload_channel +
             receive_side +
             static_cast<int>(epoch % 2) * stride_dim * stride_phase;
    }
  };

  struct PhaseExchangeResult {
    std::vector<int> send_counts;
    std::vector<int> recv_counts;
    std::vector<int> send_bytes;
    std::vector<int> recv_bytes;
  };

  void convert1StageTo3StageNeighbours(
      const std::vector<CommunicationPartner> &commPartners,
      std::vector<std::vector<CommunicationPartner>> &neighbours,
      rbmd::Real cutoffRadius);
  void ValidateGhostCutoffState(const LinkedCell *linked_cell,
                                const char *context) const;

  std::shared_ptr<DeviceData> _device_data;
  thrust::device_vector<int> leaving_flags{}; // TODO 这个应该可以放在外面公用
  thrust::device_vector<int> halo_flags{};
  thrust::device_vector<int> _staged_leaving_marks{};
  std::shared_ptr<LinkedCell> _linked_cell;

  PhaseExchangeResult exchangePhaseP2P(unsigned short d, Phase phase,
                                       MessageType msgType,
                                       LinkedCell *moleculeContainer,
                                       DomainDecomposition *domainDecomp,
                                       std::vector<int> *tracked_send_gid_counts = nullptr,
                                       std::vector<std::vector<std::pair<int, int>>> *
                                           tracked_send_gid_peers = nullptr);

  void packPhaseBuffers(unsigned short d, Phase phase, MessageType msgType,
                        LinkedCell *moleculeContainer,
                        DomainDecomposition *domainDecomp,
                        std::vector<int> &sendCounts,
                        std::vector<int> &sendBytes);

  void allocateRecvBuffers(unsigned short d, Phase phase,
                           const std::vector<int> &recvBytes);

  MsgTags _tags{};
  uint32_t _epoch{0};

  struct PendingForwardPeer {
    int rank{-1};
    int receive_side{-1};
    std::vector<rbmd::Id> send_ids{};
    std::vector<rbmd::Id> recv_ids{};
    std::vector<rbmd::Real> shift_x{};
    std::vector<rbmd::Real> shift_y{};
    std::vector<rbmd::Real> shift_z{};
    std::vector<rbmd::Real> send_reference_x{};
    std::vector<rbmd::Real> send_reference_y{};
    std::vector<rbmd::Real> send_reference_z{};
    std::vector<rbmd::Real> recv_reference_x{};
    std::vector<rbmd::Real> recv_reference_y{};
    std::vector<rbmd::Real> recv_reference_z{};
    std::vector<unsigned char> recv_retained{};
  };

  struct ForwardPeer {
    int rank{-1};
    int receive_side{-1};
    thrust::device_vector<rbmd::Id> send_indices{};
    thrust::device_vector<rbmd::Id> recv_indices{};
    thrust::device_vector<rbmd::Real> shift_x{};
    thrust::device_vector<rbmd::Real> shift_y{};
    thrust::device_vector<rbmd::Real> shift_z{};
    thrust::device_vector<rbmd::Real> send_values{};
    thrust::device_vector<rbmd::Real> recv_values{};
  };

  void CaptureGhostPhase(unsigned short d,
                         const PhaseExchangeResult& result,
                         LinkedCell* moleculeContainer,
                         DomainDecomposition* domainDecomp);
  void MarkRetainedGhosts(unsigned short d, int partner_index,
                          rbmd::Id old_total, rbmd::Id new_total);

  bool _capture_forward_coordinates{false};
  bool _pending_forward_valid{false};
  bool _forward_coordinates_valid{false};
  std::vector<std::vector<PendingForwardPeer>> _pending_forward_stages{};
  std::vector<std::vector<ForwardPeer>> _forward_stages{};

  // 用于 tag 唯一性检查的诊断信息
  std::set<std::tuple<unsigned short, Phase, uint32_t, int, int>> _used_tags{};

public:
  explicit IndirectNeighbourCommunicationScheme(ZonalMethod *zonalMethod)
      : BaseNeighbourCommunicationScheme(3, zonalMethod, false) {
    _linked_cell = LinkedCellLocator::GetInstance().GetLinkedCell();
    this->_device_data = DataManager::getInstance().getDeviceData();
    //    this->leaving_flags.resize(
    //        _linked_cell->_total_atoms_num);  // total 是动态变化的
    //    this->halo_flags.resize(_linked_cell->_total_atoms_num);
  }
  ~IndirectNeighbourCommunicationScheme() override = default;

  void exchangeMoleculesMPI(LinkedCell *moleculeContainer,
                            DomainDecomposition *domainDecomp) override;

  bool PrepareGhostCoordinateExchange(LinkedCell* moleculeContainer);
  void ForwardGhostCoordinates(LinkedCell* moleculeContainer,
                               DomainDecomposition* domainDecomp);
  void ForwardGhostFields(ForwardGhostState state,
                          LinkedCell* moleculeContainer,
                          DomainDecomposition* domainDecomp);
  void InvalidateGhostCoordinateExchange();
  uint32_t ForwardGhostPlanEpoch() const { return _epoch; }

  void initCommunicationPartners(rbmd::Real cutoffRadius,
                                 DomainDecomposition *domainDecomp) override;

  std::vector<int> get3StageNeighbourRanks() override {
    std::vector<int> neighbourRanks;
    for (auto &_fullShellNeighbour : _fullShellNeighbours) {
      if (_fullShellNeighbour.isFaceCommunicator()) {
        neighbourRanks.push_back(_fullShellNeighbour.getRank());
      }
    }
    return neighbourRanks;
  }

  void prepareNonBlockingStageImpl(LinkedCell *moleculeContainer,
                                   unsigned int stageNumber,
                                   MessageType msgType,
                                   bool removeRecvDuplicates,
                                   DomainDecomposition *domainDecomp) override;

  void finishNonBlockingStageImpl(LinkedCell *moleculeContainer,
                                  unsigned int stageNumber, MessageType msgType,
                                  bool removeRecvDuplicates,
                                  DomainDecomposition *domainDecomp) override;
};
