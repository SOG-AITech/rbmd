#pragma once

#include "../common/device_types.h"
#include "types.h"
#include "model/box.h"
namespace op {

    template <typename DEVICE>
    struct ShakeAOp {
      void operator()(const rbmd::Id num_angle, 
                      const rbmd::Real dt,
                      const rbmd::Real fmt2v, 
                      Box box,
                      const rbmd::Id* atom_id_to_idx,
                      const rbmd::Real* mass, 
                      const rbmd::Id* atoms_type,
                      const Id3* angle_id_vec,
                      rbmd::Real* shake_px,
                      rbmd::Real* shake_py,
                      rbmd::Real* shake_pz,
                      rbmd::Real* shake_vx,
                      rbmd::Real* shake_vy,
                      rbmd::Real* shake_vz,
                      const rbmd::Real* fx,
                      const rbmd::Real* fy, 
                      const rbmd::Real* fz, 
                      rbmd::Id* flag_px,
                      rbmd::Id* flag_py, 
                      rbmd::Id* flag_pz);
    };

    template <typename DEVICE>
    struct ShakeBOp {
        void operator()(rbmd::Id num_angle,
                        rbmd::Real dt,
                        rbmd::Real fmt2v,
                        Box box,
                        const rbmd::Id* atom_id_to_idx,
                        const rbmd::Real* mass,
                        const rbmd::Id* atoms_type,
                        const Id3* angle_id_vec,
                        const rbmd::Real* px,
                        const rbmd::Real* py,
                        const rbmd::Real* pz,
                        rbmd::Real* shake_vx,
                        rbmd::Real* shake_vy,
                        rbmd::Real* shake_vz,
                        const rbmd::Real* fx,
                        const rbmd::Real* fy,
                        const rbmd::Real* fz);
    };

    template <typename DEVICE>
    struct BuildShakeClustersOp {
      void operator()(rbmd::Id native_atoms,
                      rbmd::Id total_atoms,
                      int angle_per_atom,
                      const int* num_angle,
                      const rbmd::Id* angle_atom0,
                      const rbmd::Id* angle_atom1,
                      const rbmd::Id* angle_atom2,
                      const rbmd::Id* atoms_id,
                      const rbmd::Real* px,
                      const rbmd::Real* py,
                      const rbmd::Real* pz,
                      const rbmd::Id* sorted_gid,
                      const rbmd::Id* sorted_idx,
                      rbmd::Id* cluster_idx0,
                      rbmd::Id* cluster_idx1,
                      rbmd::Id* cluster_idx2,
                      int* cluster_count,
                      int* invalid_count,
                      rbmd::Id* first_invalid_gids);
    };

    template <typename DEVICE>
    struct ShakeAReplicatedOp {
      void operator()(rbmd::Id num_cluster,
                      rbmd::Id native_atoms,
                      rbmd::Real dt,
                      Box box,
                      const rbmd::Real* mass,
                      const rbmd::Id* atoms_type,
                      const rbmd::Id* cluster_idx0,
                      const rbmd::Id* cluster_idx1,
                      const rbmd::Id* cluster_idx2,
                      const rbmd::Real* shake_px,
                      const rbmd::Real* shake_py,
                      const rbmd::Real* shake_pz,
                      const rbmd::Real* px,
                      const rbmd::Real* py,
                      const rbmd::Real* pz,
                      const rbmd::Real* vx,
                      const rbmd::Real* vy,
                      const rbmd::Real* vz,
                      rbmd::Real* shake_dx,
                      rbmd::Real* shake_dy,
                      rbmd::Real* shake_dz,
                      rbmd::Real* shake_dvx,
                      rbmd::Real* shake_dvy,
                      rbmd::Real* shake_dvz);
    };

    template <typename DEVICE>
    struct ShakeBReplicatedOp {
      void operator()(rbmd::Id num_cluster,
                      rbmd::Id native_atoms,
                      Box box,
                      const rbmd::Real* mass,
                      const rbmd::Id* atoms_type,
                      const rbmd::Id* cluster_idx0,
                      const rbmd::Id* cluster_idx1,
                      const rbmd::Id* cluster_idx2,
                      const rbmd::Real* px,
                      const rbmd::Real* py,
                      const rbmd::Real* pz,
                      const rbmd::Real* vx,
                      const rbmd::Real* vy,
                      const rbmd::Real* vz,
                      rbmd::Real* shake_dvx,
                      rbmd::Real* shake_dvy,
                      rbmd::Real* shake_dvz);
    };

    template <typename DEVICE>
    struct ApplyShakeCorrectionsOp {
      void operator()(rbmd::Id num_atoms,
                      Box box,
                      bool apply_position,
                      const rbmd::Real* shake_dx,
                      const rbmd::Real* shake_dy,
                      const rbmd::Real* shake_dz,
                      const rbmd::Real* shake_dvx,
                      const rbmd::Real* shake_dvy,
                      const rbmd::Real* shake_dvz,
                      rbmd::Real* px,
                      rbmd::Real* py,
                      rbmd::Real* pz,
                      rbmd::Real* vx,
                      rbmd::Real* vy,
                      rbmd::Real* vz,
                      rbmd::Id* flag_px,
                      rbmd::Id* flag_py,
                      rbmd::Id* flag_pz);
    };

    template <>
    struct ShakeAOp<device::DEVICE_GPU> {
      void operator()(rbmd::Id num_angle,
                      rbmd::Real dt,
                      rbmd::Real fmt2v,
                      Box box,
                      const rbmd::Id* atom_id_to_idx,
                      const rbmd::Real* mass,
                      const rbmd::Id* atoms_type,
                      const Id3* angle_id_vec,
                      rbmd::Real* shake_px,
                      rbmd::Real* shake_py,
                      rbmd::Real* shake_pz,
                      rbmd::Real* shake_vx,
                      rbmd::Real* shake_vy,
                      rbmd::Real* shake_vz,
                      const rbmd::Real* fx,
                      const rbmd::Real* fy,
                      const rbmd::Real* fz,
                      rbmd::Id* flag_px,
                      rbmd::Id* flag_py,
                      rbmd::Id* flag_pz);
    };

    template <>
    struct ShakeBOp<device::DEVICE_GPU>  {
        void operator()(const rbmd::Id num_angle,
                        const rbmd::Real dt,
                        const rbmd::Real fmt2v,
                        Box box,
                        const rbmd::Id* atom_id_to_idx,
                        const rbmd::Real* mass,
                        const rbmd::Id* atoms_type,
                        const Id3* angle_id_vec,
                        const rbmd::Real* px,
                        const rbmd::Real* py,
                        const rbmd::Real* pz,
                        rbmd::Real* shake_vx,
                        rbmd::Real* shake_vy,
                        rbmd::Real* shake_vz,
                        const rbmd::Real* fx,
                        const rbmd::Real* fy,
                        const rbmd::Real* fz);
    };

    template <>
    struct BuildShakeClustersOp<device::DEVICE_GPU> {
      void operator()(rbmd::Id native_atoms,
                      rbmd::Id total_atoms,
                      int angle_per_atom,
                      const int* num_angle,
                      const rbmd::Id* angle_atom0,
                      const rbmd::Id* angle_atom1,
                      const rbmd::Id* angle_atom2,
                      const rbmd::Id* atoms_id,
                      const rbmd::Real* px,
                      const rbmd::Real* py,
                      const rbmd::Real* pz,
                      const rbmd::Id* sorted_gid,
                      const rbmd::Id* sorted_idx,
                      rbmd::Id* cluster_idx0,
                      rbmd::Id* cluster_idx1,
                      rbmd::Id* cluster_idx2,
                      int* cluster_count,
                      int* invalid_count,
                      rbmd::Id* first_invalid_gids);
    };

    template <>
    struct ShakeAReplicatedOp<device::DEVICE_GPU> {
      void operator()(rbmd::Id num_cluster,
                      rbmd::Id native_atoms,
                      rbmd::Real dt,
                      Box box,
                      const rbmd::Real* mass,
                      const rbmd::Id* atoms_type,
                      const rbmd::Id* cluster_idx0,
                      const rbmd::Id* cluster_idx1,
                      const rbmd::Id* cluster_idx2,
                      const rbmd::Real* shake_px,
                      const rbmd::Real* shake_py,
                      const rbmd::Real* shake_pz,
                      const rbmd::Real* px,
                      const rbmd::Real* py,
                      const rbmd::Real* pz,
                      const rbmd::Real* vx,
                      const rbmd::Real* vy,
                      const rbmd::Real* vz,
                      rbmd::Real* shake_dx,
                      rbmd::Real* shake_dy,
                      rbmd::Real* shake_dz,
                      rbmd::Real* shake_dvx,
                      rbmd::Real* shake_dvy,
                      rbmd::Real* shake_dvz);
    };

    template <>
    struct ShakeBReplicatedOp<device::DEVICE_GPU> {
      void operator()(rbmd::Id num_cluster,
                      rbmd::Id native_atoms,
                      Box box,
                      const rbmd::Real* mass,
                      const rbmd::Id* atoms_type,
                      const rbmd::Id* cluster_idx0,
                      const rbmd::Id* cluster_idx1,
                      const rbmd::Id* cluster_idx2,
                      const rbmd::Real* px,
                      const rbmd::Real* py,
                      const rbmd::Real* pz,
                      const rbmd::Real* vx,
                      const rbmd::Real* vy,
                      const rbmd::Real* vz,
                      rbmd::Real* shake_dvx,
                      rbmd::Real* shake_dvy,
                      rbmd::Real* shake_dvz);
    };

    template <>
    struct ApplyShakeCorrectionsOp<device::DEVICE_GPU> {
      void operator()(rbmd::Id num_atoms,
                      Box box,
                      bool apply_position,
                      const rbmd::Real* shake_dx,
                      const rbmd::Real* shake_dy,
                      const rbmd::Real* shake_dz,
                      const rbmd::Real* shake_dvx,
                      const rbmd::Real* shake_dvy,
                      const rbmd::Real* shake_dvz,
                      rbmd::Real* px,
                      rbmd::Real* py,
                      rbmd::Real* pz,
                      rbmd::Real* vx,
                      rbmd::Real* vy,
                      rbmd::Real* vz,
                      rbmd::Id* flag_px,
                      rbmd::Id* flag_py,
                      rbmd::Id* flag_pz);
    };
}  // namespace op
