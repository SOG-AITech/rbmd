#pragma once

// 定义数据类型枚举
enum class DataType {
  FLOAT,
  DOUBLE,
  INT,
  // 其他类型...
};

// 定义操作类型枚举
enum class OpType {
  SUM,
  // 其他操作...
};

class ICommunicator {
 public:
  virtual ~ICommunicator() = default;

  virtual void AllReduce(const void* sendbuf, void* recvbuf, int count,
                         DataType data_type, OpType op) = 0;

  virtual void Reduce(const void* sendbuf, void* recvbuf, int count,
                      DataType data_type, OpType op, int root) = 0;

  virtual void BroadCast(void* buffer, int count, DataType data_type,
                         int root) = 0;

  virtual void AllGather(const void* sendbuf, int send_count,
                         DataType send_data_type, void* recvbuf, int recv_count,
                         DataType recv_data_type) = 0;
};