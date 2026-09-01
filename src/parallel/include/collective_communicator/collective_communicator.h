#pragma once
#include "i_communicator.h"


#include <memory>

class CollectiveCommunicator {
public:
  enum class Backend {
    MPI,
    NCCL,
    //OneCCL  本机环境有点小问题
  };

  CollectiveCommunicator(Backend backend, int rank, int size);

  ~CollectiveCommunicator() = default;

  void AllReduce(const void* sendbuf, void* recvbuf, int count, DataType dtype,
                 OpType op) const;
  void Reduce(const void* sendbuf, void* recvbuf, int count, DataType data_type,
            OpType op, int root) const;
  void BroadCast(void* buffer, int count, DataType data_type,
                 int root) const;
  void AllGather(const void* sendbuf, int send_count, DataType send_data_type,
                 void* recvbuf, int recv_count,
                 DataType recv_data_type) const;
private:
  std::unique_ptr<ICommunicator> communicator;
};