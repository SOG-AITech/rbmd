#pragma once
#include <thrust/device_vector.h>
#include "../halo_leaving_atoms.h"
//! 借助thruts来实现
// TODO 分析和halodata的关系  recvbuf不需要吧，直接是填充到halodata里面去？

class CommunicationBuffer {
 public:
  CommunicationBuffer();

  ~CommunicationBuffer() = default;

  // Device vectors to store Halo particles
  //thrust::device_vector<HaloAtom> _halo_atoms;
  thrust::device_vector<char> _ghost_atoms;   // id type px py pz [charge]
  thrust::device_vector<char> _shake_forward_atoms;
  thrust::device_vector<char> _shake_reverse_atoms;

  // Device vectors to store Leaving particles (exchange buffer)
  // 对齐 LAMMPS：使用 double buffer 存储 pack 后的原子数据（包含拓扑）
  // Buffer layout: [atom0_data (size_exchange doubles)] [atom1_data] ...
  thrust::device_vector<double> _leaving_atoms_double; // LAMMPS layout exchange buffer
  
  // 保留旧的 char buffer 以兼容性（如有需要）
  thrust::device_vector<char> _leaving_atoms; // 废弃：仅为兼容性保留


  void ResizeLeaving(unsigned int leaving_size);

  void ResizeGhost(unsigned int halo_size);

  void ResizeLeavingByByteSize(unsigned long leaving_size);

  void ResizeGhostByByteSize(unsigned long halo_size);
  void ResizeShakeForwardByByteSize(unsigned long byte_size);
  void ResizeShakeReverseByByteSize(unsigned long byte_size);

  /**
   * @brief 使用 LAMMPS layout 调整 leaving buffer 大小
   * 
   * @param num_atoms 要发送的原子数量
   * @param size_exchange 每个原子的 buffer 大小（单位：double）
   */
  void ResizeLeavingExchange(unsigned int num_atoms, int size_exchange);

  void Clear();
  
  // ========== 基础 buffer 大小（旧模式，字节计数） ==========
  size_t _leaving_size = 0;  // 废弃：仅为兼容性保留
  size_t _ghost_size = 0;
  size_t _shake_forward_size = 0;
  size_t _shake_reverse_size = 0;
  
  // ========== LAMMPS layout 配置（对齐 LAMMPS size_exchange） ==========
  int _size_exchange = 0;      // 每个原子在 exchange buffer 中的大小（单位：double）
  
  // 拓扑配置参数（从 DeviceData 获取）
  int _bond_per_atom = 0;
  int _angle_per_atom = 0;
  int _dihedral_per_atom = 0;
  int _improper_per_atom = 0;
  int _maxspecial = 0;
  
  // 清理MPI类型
};
