#pragma once

#include <mpi.h>

#include "common/types.h"


// 一个辅助结构体，用于临时存储要发送给其他进程的原子数据
struct AtomicAtomDataPacket {
  rbmd::Id atom_id;
  rbmd::Id atom_type;
  rbmd::Real px;
  rbmd::Real py;
  rbmd::Real pz;
  rbmd::Real vx;
  rbmd::Real vy;
  rbmd::Real vz;
};


struct ChargeAtomDataPacket:AtomicAtomDataPacket {
  rbmd::Real charge;
};

struct FullAtomDataPacket:ChargeAtomDataPacket {
  rbmd::Id molecules_id;
  int image_x;
  int image_y;
  int image_z;
};
void CreateMpiAtomicAtomDataPacket(MPI_Datatype* new_type);
void CreateMpiChargeAtomDataPacket(MPI_Datatype* new_type);
void CreateMpiFullAtomDataPacket(MPI_Datatype* new_type);
