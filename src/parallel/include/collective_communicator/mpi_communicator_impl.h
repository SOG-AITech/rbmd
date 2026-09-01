#pragma once
#include "mpi.h"
#include "i_communicator.h"

class MpiCommunicatorImpl : public ICommunicator {
public:
  MpiCommunicatorImpl();
  ~MpiCommunicatorImpl() override = default;

  void AllReduce(const void* sendbuf, void* recvbuf, int count,
                 DataType data_type, OpType op) override;
  void Reduce(const void* sendbuf, void* recvbuf, int count, DataType data_type,
              OpType op, int root) override;
  void BroadCast(void* buffer, int count, DataType data_type,
                 int root) override;
  void AllGather(const void* sendbuf, int send_count, DataType send_data_type,
                 void* recvbuf, int recv_count,
                 DataType recv_data_type) override;
  MPI_Datatype GetMpiDataType(DataType data_type);
  MPI_Op GetMpiOp(OpType op);

 private:
  MPI_Comm _mpi_comm = MPI_COMM_WORLD;
};
