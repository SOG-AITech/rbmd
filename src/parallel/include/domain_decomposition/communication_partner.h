#pragma once
#include "mpi.h"

#include <cstddef>
#include <vector>

#include <thrust/host_vector.h>

#include "common/rbmd_define.h"
#include "communication_buffer.h"
#include "neighbor_list/include/linked_cell/linked_cell.h"

typedef enum {
  LEAVING_AND_HALO_COPIES = 0, /** send process-leaving particles and
                                  halo-copies together in one message */
  HALO_COPIES = 1,             /** send halo-copies only */
  LEAVING_ONLY = 2,            /** send process-leaving particles only */
  FORCES = 3                   /** send forces */
} MessageType;

struct PositionInfo {
  rbmd::Real _bothLow[3], _bothHigh[3];
  rbmd::Real _leavingLow[3], _leavingHigh[3];
  rbmd::Real _copiesLow[3], _copiesHigh[3];
  rbmd::Real _shift[3];  //! for periodic boundaries
  int _offset[3];
  bool _enlarged[3][2];
};

/**
 * (Bi-Directional) MPI Communication Partner.
 */
class CommunicationPartner {
 public:
  CommunicationPartner(int r, const rbmd::Real hLo[3], const rbmd::Real hHi[3],
                       const rbmd::Real bLo[3], const rbmd::Real bHi[3],
                       const rbmd::Real sh[3], const int offset[3],
                       const bool enlarged[3][2]);
  explicit CommunicationPartner(int r);
  CommunicationPartner(int r, const rbmd::Real leavingLo[3],
                       const rbmd::Real leavingHi[3]);

  CommunicationPartner(const CommunicationPartner& o);

  CommunicationPartner() = delete;

  CommunicationPartner& operator=(const CommunicationPartner& b);

  ~CommunicationPartner();

  // 初始化发送操作（收集数据并启动异步发送）
  // end_atom_idx参数用于三阶段通信，指定要处理的原子范围（见CollectMoleculesInRegion）
  void initSend(LinkedCell* moleculeContainer, const MPI_Comm& comm, Box* box,
                thrust::device_vector<int>& leaving_flags,
                thrust::device_vector<int>& halo_flags, MessageType msgType,
                rbmd::Id start_atom_idx = 0, rbmd::Id end_atom_idx = 0,
                thrust::device_vector<int>* staged_leaving_marks = nullptr);

  bool testSend();

  // 准备消息（收集数据但不发送）
  // end_atom_idx参数用于三阶段通信，指定要处理的原子范围（见CollectMoleculesInRegion）
  void PrepareMessage(LinkedCell* moleculeContainer, Box* box,
                      thrust::device_vector<int>& leaving_flags,
                      thrust::device_vector<int>& halo_flags,
                      MessageType msgType, const MPI_Comm& comm = MPI_COMM_WORLD,
                      int current_rank = -1, int total_rank = -1,
                      rbmd::Id start_atom_idx = 0, rbmd::Id end_atom_idx = 0,
                      thrust::device_vector<int>* staged_leaving_marks = nullptr);

  int GetSendCount(MessageType msgType) const;

  size_t GetSendBytes(MessageType msgType) const;

  char* GetSendBufferPtr(MessageType msgType);

  void EnsureRecvBuffer(MessageType msgType, size_t recvBytes);

  char* GetRecvBufferPtr(MessageType msgType);

  void ClearSendBuffer();

  void ResetReceive();

  void PrepareShakeForwardMessage(const thrust::host_vector<rbmd::Id>& atom_ids,
                                  const thrust::host_vector<rbmd::Real>& values);
  void PrepareShakeReverseMessage(const thrust::host_vector<rbmd::Id>& atom_ids,
                                  const thrust::host_vector<rbmd::Real>& values);
  void ProcessShakeForwardData();
  void ProcessShakeReverseData();

  /**
     * @brief 探测指定TAG的消息，如果探测到，则发出非阻塞接收(Irecv)。
     * 这个函数是幂等的，对于同一个TAG，它只会发出一次Irecv。
     * @param comm MPI通信域
     * @param tag 要探测的消息标签 (LEAVING_TAG 或 HALO_TAG)
     */
  void iprobeAndPostRecv(MPI_Comm comm, int tag);

  bool testRecv(LinkedCell *linked_cell,
                bool force = false);

  void initRecv(int numParticles, const MPI_Comm& comm,
                const MPI_Datatype& type);

  void deadlockDiagnosticSend() const;
  void deadlockDiagnosticRecv() const;
  void deadlockDiagnosticSendRecv() const;

  int getRank() const { return _rank; }

  const int* getOffset() { return _haloInfo[0]._offset; }

  //! Specifies, whether the communication to the CommunicationPartner is along
  //! a shared face (_offset has only one entry != 0)
  //! @return returns whether they are direct face sharing neighbours
  bool isFaceCommunicator() const {
    return (!!_haloInfo[0]._offset[0] + !!_haloInfo[0]._offset[1] +
            !!_haloInfo[0]._offset[2]) == 1;
  }
  //! @return returns in which direction the face is shared. If it is not a face
  //! communicator, -1 is returned
  int getFaceCommunicationDirection() const {
    if (!isFaceCommunicator()) return -1;
    return !!_haloInfo[0]._offset[1] * 1 + !!_haloInfo[0]._offset[2] * 2;
  }

  void enlargeInOtherDirections(unsigned int d, rbmd::Real enlargement) {
    for (unsigned int p = 0; p < _haloInfo.size(); p++) {
      for (unsigned int d2 = 0; d2 < 3; d2++) {
        if (d2 == d) continue;
        if (!_haloInfo[p]._enlarged[d2][0]) {
          _haloInfo[p]._bothLow[d2] -= enlargement;
          _haloInfo[p]._leavingLow[d2] -= enlargement;
          _haloInfo[p]._copiesLow[d2] -= enlargement;
          _haloInfo[p]._enlarged[d2][0] = true;
        }
        if (!_haloInfo[p]._enlarged[d2][1]) {
          _haloInfo[p]._bothHigh[d2] += enlargement;
          _haloInfo[p]._leavingHigh[d2] += enlargement;
          _haloInfo[p]._copiesHigh[d2] += enlargement;
          _haloInfo[p]._enlarged[d2][1] = true;
        }
      }
    }
  }

  void ExtendLeavingAlongFaceDirection(unsigned int dim,
                                       rbmd::Real local_extent) {
    if (dim >= 3 || local_extent <= rbmd::Real(0)) {
      return;
    }
    for (auto& info : _haloInfo) {
      const int face_offset = info._offset[dim];
      if (face_offset == 0) {
        continue;
      }
      if (face_offset > 0) {
        info._leavingHigh[dim] = info._leavingLow[dim] + local_extent;
      } else {
        info._leavingLow[dim] = info._leavingHigh[dim] - local_extent;
      }
      info._bothLow[dim] = MIN(info._leavingLow[dim], info._copiesLow[dim]);
      info._bothHigh[dim] = MAX(info._leavingHigh[dim], info._copiesHigh[dim]);
    }
  }

  void ExtendLeavingInOtherDirections(
      unsigned int dim, const std::array<rbmd::Real, 3>& local_extents) {
    if (dim >= 3) {
      return;
    }
    for (auto& info : _haloInfo) {
      for (unsigned int d2 = 0; d2 < 3; ++d2) {
        if (d2 == dim) continue;
        const rbmd::Real extent = local_extents[d2];
        if (extent <= rbmd::Real(0)) {
          continue;
        }
        info._leavingLow[d2] -= extent;
        info._leavingHigh[d2] += extent;
        info._bothLow[d2] = MIN(info._leavingLow[d2], info._copiesLow[d2]);
        info._bothHigh[d2] = MAX(info._leavingHigh[d2], info._copiesHigh[d2]);
      }
    }
  }


  //! Combines current CommunicationPartner with the given partner
  //! @param partner which to add to the current CommunicationPartner
  void add(CommunicationPartner partner);

  size_t getDynamicSize();

  void print(std::ostream& stream) const;

  void ProcessLeavingData(LinkedCell* linked_cell);
  void ProcessGhostData(LinkedCell* linked_cell);
  void ResetRecvStatus();
  
  // 输出接收到的原子详细信息到CSV文件
  void OutputLeavingAtomsToCSV(const std::string& filename, int current_rank, rbmd::Id timestep = 0);
  void OutputGhostAtomsToCSV(const std::string& filename, int current_rank, rbmd::Id timestep = 0);
  
  // 输出每个进程的haloinfo到同一个txt文件
  void outputHaloInfoToFile(int current_rank, int total_rank, const MPI_Comm& comm = MPI_COMM_WORLD);

  // 根据文档3.2.3节的三阶段通信方案收集需要发送的原子
  // 参数说明：
  // - start_atom_idx: 始终为0（从第一个原子开始）
  // - end_atom_idx: 当前阶段要处理的原子总数（包括native和前面阶段累积的halo）
  //   * 对于LEAVING_ONLY：使用默认值0，函数内部使用native_num
  //   * 对于HALO_COPIES（三阶段）：传递当前_total_atoms_num
  //     - 第1阶段（x维）：end_atom_idx = native_num
  //     - 第2阶段（y维）：end_atom_idx = native_num + x维halo数
  //     - 第3阶段（z维）：end_atom_idx = native_num + x维halo + y维halo数
  // 注意：该方法必须是public，因为它内部包含__device__ lambda，
  // CUDA编译器不允许在private/protected方法中使用扩展__device__ lambda
  void CollectMoleculesInRegion(LinkedCell* linked_cell,
                                std::vector<PositionInfo>& position_infos,
                                Box* box,
                                thrust::device_vector<int>& leaving_flags,
                                thrust::device_vector<int>& halo_flags,
                                MessageType msgType,
                                rbmd::Id start_atom_idx,
                                rbmd::Id end_atom_idx,
                                thrust::device_vector<int>* staged_leaving_marks);

 private:
  std::shared_ptr<DeviceData> _device_data;
  enum HaloOrLeavingCorrection { HALO, LEAVING, NONE, FORCES };


  int _rank;
  int _countLeavingTested = 0;
  int _countHaloTested = 0;
  bool _leavingReceivedDone = false;
  bool _haloReceivedDone = false;
  bool _isReceiving = false;
  std::vector<PositionInfo> _haloInfo;

  // -- 发送相关 --
  MPI_Request _sendLeavingRequest;
  MPI_Request _sendHaloRequest;
  MPI_Status _sendLeavingStatus;
  MPI_Status _sendHaloStatus;
  bool _msgSent;
  bool _isSending;

  // -- 接收相关 --
  // 为 Leaving 和 Halo 消息提供独立的接收缓冲区和状态
  // Leaving 粒子接收状态
  MPI_Request _recvLeavingRequest;
  bool _isLeavingRecvPosted;      // 是否已为Leaving发出Irecv
  bool _isLeavingRecvFinalized;   // 是否已完成Leaving接收并解包

  // Halo 粒子接收状态
  MPI_Request _recvHaloRequest;
  bool _isHaloRecvPosted;         // 是否已为Halo发出Irecv
  bool _isHaloRecvFinalized;      // 是否已完成Halo接收并解包
  CommunicationBuffer _send_buf{}, _recv_buf{};
  // CommunicationBuffer _sendBuf, _recvBuf; // used to be ParticleData and TODO
  // force

  // void collectLeavingMoleculesFromInvalidParticles(std::vector<Atom>&
  // invalidParticles, double lowCorner [3], double highCorner [3], double shift
  // [3]);   //TODO FOR DCU
};

// 在主机代码中
struct IsMarked {
  __host__ __device__ bool operator()(const int flag) const {
    return flag == 1;
  }
};
