#include "common/rbmd_define.h"
#include "gpu_device_binding.h"
#include "mpi_communicator_impl.h"

#define  USE_GPU

MpiCommunicatorImpl::MpiCommunicatorImpl() {
#ifdef USE_GPU
  int current_rank, total_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
  rbmd::BindGpuDeviceForRankOrAbort("MpiCommunicatorImpl", current_rank,
                                    total_ranks);
#else
  return;
#endif
}

void MpiCommunicatorImpl::AllReduce(const void* sendbuf, void* recvbuf,
                                    int count, DataType data_type, OpType op) {
  MPI_CHECK(MPI_Allreduce(sendbuf, recvbuf, count, GetMpiDataType(data_type),
    GetMpiOp(op), _mpi_comm));
}

void MpiCommunicatorImpl::Reduce(const void* sendbuf, void* recvbuf, int count,
                                 DataType data_type, OpType op, int root) {
  MPI_CHECK(MPI_Reduce(sendbuf, recvbuf, count, GetMpiDataType(data_type),
    GetMpiOp(op), root, _mpi_comm));
}

void MpiCommunicatorImpl::BroadCast(void* buffer, int count, DataType data_type,
                                    int root) {
  MPI_CHECK(
      MPI_Bcast(buffer, count, GetMpiDataType(data_type), root, _mpi_comm));
}

void MpiCommunicatorImpl::AllGather(const void* sendbuf, int send_count,
                                    DataType send_data_type, void* recvbuf,
                                    int recv_count, DataType recv_data_type) {
  MPI_CHECK(MPI_Allgather(sendbuf, send_count, GetMpiDataType(send_data_type),
    recvbuf, recv_count, GetMpiDataType(recv_data_type),
    _mpi_comm));
}

MPI_Datatype MpiCommunicatorImpl::GetMpiDataType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT:
      return MPI_FLOAT; // TODO 用宏
    case DataType::INT:
      return MPI_INT;
    // 处理其他类型...
    default:
      return MPI_FLOAT;
  }
}

MPI_Op MpiCommunicatorImpl::GetMpiOp(OpType op) {
  switch (op) {
    case OpType::SUM:
      return MPI_SUM;
    // 处理其他操作...
    default:
      return MPI_SUM;
  }
}
