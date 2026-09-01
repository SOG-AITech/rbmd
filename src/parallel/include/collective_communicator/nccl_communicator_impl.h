// NcclCommunicatorImpl.h
#pragma once
#include "mpi.h"
#include "common/rbmd_define.h"
#include <iostream>

#include "i_communicator.h"

#if USE_CCL
class NcclCommunicatorImpl : public ICommunicator {
 public:
  NcclCommunicatorImpl(int current_rank, int total_ranks);
  ~NcclCommunicatorImpl() override;

  void AllReduce(const void* sendbuf, void* recvbuf, int count,
                 DataType data_type, OpType op) override;
  void Reduce(const void* sendbuf, void* recvbuf, int count, DataType data_type,
              OpType op, int root) override;
  void BroadCast(void* buffer, int count, DataType data_type,
                 int root) override;
  void AllGather(const void* sendbuf, int send_count, DataType send_data_type,
                 void* recvbuf, int recv_count,
                 DataType recv_data_type) override;

 private:
  ncclUniqueId _nccl_id{};
  /// nccl通信器
  ncclComm_t _nccl_comm{};
  STREAM_T _stream{};  // todo 确定是否要额外的stream

  int _current_rank{};
  int _total_ranks{};

  ncclRedOp_t GetNcclOp(OpType op);
  ncclDataType_t GetNcclDataType(DataType data_type);
};
#endif
