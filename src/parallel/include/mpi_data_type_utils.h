#pragma once

#include "mpi.h"
#include <thrust/tuple.h>
#include "common/rbmd_define.h"
#include "halo_leaving_atoms.h"

struct Tuple2LeavingAtom {
  __host__ __device__ LeavingAtom operator()(
      const thrust::tuple<rbmd::Id, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real>& t)
      const {
    LeavingAtom p{};
    p.id = thrust::get<0>(t);
    p.rx = thrust::get<1>(t);
    p.ry = thrust::get<2>(t);
    p.rz = thrust::get<3>(t);
    p.vx = thrust::get<4>(t);
    p.vy = thrust::get<5>(t);
    p.vz = thrust::get<6>(t);
    p.charge = thrust::get<7>(t);   //缺少原子类型，短程不用。没有电荷就填充0，这个麻烦程度不高
    return p;
  }
};

struct Tuple2HaloAtom {
  __host__ __device__ HaloAtom
  operator()(const thrust::tuple<rbmd::Real, rbmd::Real, rbmd::Real>& t) const {
    HaloAtom p{};
    p.rx = thrust::get<0>(t);
    p.ry = thrust::get<1>(t);
    p.rz = thrust::get<2>(t);
    return p;
  }
};

struct LeavingAtom2Tuple {
  __host__ __device__
      thrust::tuple<rbmd::Id, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real, rbmd::Real>
      operator()(const LeavingAtom& p) const {
    return thrust::make_tuple(p.id, p.rx, p.ry, p.rz, p.vx, p.vy, p.vz,
                              p.charge);
  }
};

struct HaloAtom2Tuple {
  __host__ __device__ thrust::tuple<rbmd::Real, rbmd::Real, rbmd::Real> operator()(
      const HaloAtom& p) const {
    return thrust::make_tuple(p.rx, p.ry, p.rz);
  }
};

class MpiDataTypeUtils {
 public:
  MpiDataTypeUtils(const MpiDataTypeUtils&) = delete;
  MpiDataTypeUtils& operator=(const MpiDataTypeUtils&) = delete;

  static MpiDataTypeUtils& GetInstance() {
    static MpiDataTypeUtils instance;
    return instance;
  }

  MPI_Datatype CreateLeavingAtomMpiType();
  MPI_Datatype CreateHaloAtomMpiType();

  MPI_Datatype GetLeavingAtomType();

  MPI_Datatype GetHaloAtomType();

 private:
  MpiDataTypeUtils() = default;
  ~MpiDataTypeUtils() = default;

  MPI_Datatype _leaving_atom_type = MPI_DATATYPE_NULL;
  MPI_Datatype _halo_atom_type = MPI_DATATYPE_NULL;


};
