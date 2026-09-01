#include "collective_communicator.h"
#if USE_CCL
#include "nccl_communicator_impl.h"
#endif
#include "mpi_communicator_impl.h"
#include <stdexcept>

CollectiveCommunicator::CollectiveCommunicator(Backend backend, int rank,
                                               int size) {
  switch (backend) {
    case Backend::MPI:
      communicator = std::make_unique<MpiCommunicatorImpl>();
      break;
    case Backend::NCCL:
#if USE_CCL
      communicator = std::make_unique<NcclCommunicatorImpl>(rank, size);
#else
      throw std::runtime_error("NCCL backend requires -DCCL=ON");
#endif
      break;
    // case Backend::OneCCL:
    //   communicator = std::make_unique<OnecclCommunicatorImpl>(rank, size);
    //   break;
    // 处理其他通信库...
    default:
      throw std::runtime_error("Unsupported backend");
  }
}

void CollectiveCommunicator::AllReduce(const void* sendbuf, void* recvbuf,
                                       int count, DataType dtype,
                                       OpType op) const {
  communicator->AllReduce(sendbuf, recvbuf, count, dtype, op);
}

void CollectiveCommunicator::Reduce(const void* sendbuf, void* recvbuf,
    int count, DataType data_type, OpType op, int root) const {
  communicator->Reduce(sendbuf,recvbuf,count,data_type,op,root);
}

void CollectiveCommunicator::BroadCast(void* buffer, int count,
    DataType data_type, int root) const {
  communicator->BroadCast(buffer,count,data_type,root);
}

void CollectiveCommunicator::AllGather(const void* sendbuf, int send_count,
    DataType send_data_type, void* recvbuf, int recv_count,
    DataType recv_data_type) const {
  communicator->AllGather(sendbuf,send_count,send_data_type,recvbuf,recv_count,recv_data_type);
}
