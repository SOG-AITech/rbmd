#include "common/rbmd_define.h"
#include "gpu_device_binding.h"
#include "nccl_communicator_impl.h"
#include <stdexcept>

#if USE_CCL
NcclCommunicatorImpl::NcclCommunicatorImpl(int current_rank, int total_ranks)
    : _current_rank(current_rank), _total_ranks(total_ranks) {
  // 在rank 0上获取NCCL唯一ID并广播给所有其他rank
  if (current_rank == 0) ncclGetUniqueId(&_nccl_id);
  MPI_CHECK(
      MPI_Bcast(&_nccl_id, sizeof(_nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
  // 设置进程到设备  之后的kernel和runtime都会执行到相应的设备了
  rbmd::BindGpuDeviceForRankOrAbort("NcclCommunicatorImpl", current_rank,
                                    total_ranks);
  NCCL_CHECK(
      ncclCommInitRank(&_nccl_comm, total_ranks, _nccl_id, current_rank));
  CHECK_RUNTIME(STREAM_CREATE(&_stream));

}

NcclCommunicatorImpl::~NcclCommunicatorImpl() {
  ncclCommDestroy(_nccl_comm);
  CHECK_RUNTIME(STREAM_DESTROY(_stream));
}

ncclRedOp_t NcclCommunicatorImpl::GetNcclOp(OpType op) {
  switch (op) {
    case OpType::SUM:
      return ncclSum;
    // 处理其他操作...
    default:
      return ncclSum;
  }
}

ncclDataType_t NcclCommunicatorImpl::GetNcclDataType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT:
      return ncclFloat;  // TODO 用宏
    case DataType::INT:
      return ncclInt;
    // 处理其他类型...
    default:
      return ncclFloat;
  }
}

void NcclCommunicatorImpl::AllReduce(const void* sendbuf, void* recvbuf,
                                     int count, DataType data_type, OpType op) {
  ncclAllReduce(sendbuf, recvbuf, count, GetNcclDataType(data_type),
                GetNcclOp(op), _nccl_comm, _stream);
}

void NcclCommunicatorImpl::Reduce(const void* sendbuf, void* recvbuf, int count,
                                  DataType data_type, OpType op, int root) {
  ncclReduce(sendbuf, recvbuf, count, GetNcclDataType(data_type), GetNcclOp(op),
             root, _nccl_comm, _stream);
}

void NcclCommunicatorImpl::BroadCast(void* buffer, int count,
                                     DataType data_type, int root) {
  ncclBcast(buffer, count, GetNcclDataType(data_type), root, _nccl_comm,
            _stream);
}

void NcclCommunicatorImpl::AllGather(const void* sendbuf, int send_count,
                                     DataType send_data_type, void* recvbuf,
                                     int recv_count, DataType recv_data_type) {
  if (send_data_type == recv_data_type) {
    ncclAllGather(sendbuf, recvbuf, send_count, GetNcclDataType(send_data_type),
                  _nccl_comm, _stream);
  } else {
    throw std::runtime_error("Error param in NcclCommunicatorImpl::AllGather");
  }
}
#endif
