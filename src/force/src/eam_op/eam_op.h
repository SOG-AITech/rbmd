#pragma once
#include "../../common/device_types.h"
#include "../../common/types.h"
#include "../../data_manager/include/model/box.h"
#include "../../force/include/eam.h"

namespace op {

struct EAMDirectTraversalData {
  const rbmd::Id* per_atom_cell_id;
  const rbmd::Id* cell_atom_start;
  const rbmd::Id* cell_atom_end;
  const rbmd::Id* sorted_atom_indices;
  LinkedCellDeviceDataPtr* linked_cell;
  rbmd::Id neighbor_cell_num;
  rbmd::Id cell_count_within_cutoff;
};

//Verlet:: Fp Verlet
template <typename DEVICE>
struct ComputeFpVerlet
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real cut_off,
        const rbmd::Id num_atoms,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const Real7*  rhor_spline,
        const Real7* frho_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        rbmd::Real* eam_fp,
        rbmd::Real* energy_embedding);
};

//Verlet:: EAMForceVerletOp
template <typename DEVICE>
struct ComputeEAMForceVerlet
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real cut_off,
        const rbmd::Id num_atoms,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const Real7*  rhor_spline,
        const Real7*  z2r_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        rbmd::Real* eam_fp,
        rbmd::Real* fx,
        rbmd::Real* fy,
        rbmd::Real* fz,
        rbmd::Real* energy_pair);
};

template <typename DEVICE>
struct ComputeFpDirect {
  void operator()(Box box, EAMParameters eam_paras, rbmd::Real cut_off,
                  rbmd::Id num_atoms, EAMDirectTraversalData traversal,
                  const Real7* rhor_spline, const Real7* frho_spline,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, rbmd::Real* eam_fp,
                  rbmd::Real* energy_embedding);
};

template <typename DEVICE>
struct ComputeEAMForceRBLDirect {
  void operator()(Box box, EAMParameters eam_paras, rbmd::Real r_core,
                  rbmd::Real cut_off, rbmd::Id num_atoms,
                  rbmd::Id neighbor_sample_num,
                  rbmd::Id selection_frequency,
                  EAMDirectTraversalData traversal,
                  const Real7* rhor_spline, const Real7* z2r_spline,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* eam_fp,
                  rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz);
};

template <typename DEVICE>
struct ComputeEAMEnergyDirect {
  void operator()(Box box, EAMParameters eam_paras, rbmd::Real cut_off,
                  rbmd::Id num_atoms, EAMDirectTraversalData traversal,
                  const Real7* rhor_spline, const Real7* z2r_spline,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* eam_fp,
                  rbmd::Real* energy_pair);
};

//RBL :: EAMRhoRBL
template <typename DEVICE>
struct ComputeRhoRBL
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real rs,
        const rbmd::Real rc,
        const rbmd::Id num_atoms,
        const rbmd::Id neighbor_sample_num,
        const rbmd::Id pice_num,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const rbmd::Id* id_random_neighbor,
        const rbmd::Id* random_neighbor_num,
        const Real7*  rhor_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        rbmd::Real* eam_rho);
};


template <typename DEVICE>
struct ComputeFpRBL
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real rs,
        const rbmd::Real rc,
        const rbmd::Id num_atoms,
        const rbmd::Id neighbor_sample_num,
        const rbmd::Id pice_num,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const rbmd::Id* id_random_neighbor,
        const rbmd::Id* random_neighbor_num,
        const Real7*  rhor_spline,
        const Real7* frho_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        rbmd::Real* eam_fp);
};

template <typename DEVICE>
struct ComputeEAMForceRBL
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real rs,
        const rbmd::Real rc,
        const rbmd::Id num_atoms,
        const rbmd::Id neighbor_sample_num,
        const rbmd::Id pice_num,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const rbmd::Id* id_random_neighbor,
        const rbmd::Id* random_neighbor_num,
        const Real7*  rhor_spline,
        const Real7*  z2r_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        const  rbmd::Real* eam_fp,
        rbmd::Real* fx,
        rbmd::Real* fy,
        rbmd::Real* fz);
};


template <typename DEVICE>
struct ComputeEAMEnergy
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real cut_off,
        const rbmd::Id num_atoms,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const Real7* rhor_spline,
        const Real7* z2r_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        const rbmd::Real* eam_fp,
        rbmd::Real* energy_pair);
};


//////////////////////////////////////////////
///

template <>
struct ComputeFpVerlet<device::DEVICE_GPU>
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real cut_off,
        const rbmd::Id num_atoms,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const Real7*  rhor_spline,
        const Real7* frho_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        rbmd::Real* eam_fp,
        rbmd::Real* energy_embedding);
};

template <>
struct ComputeEAMForceVerlet<device::DEVICE_GPU> {
  void operator()(Box box, EAMParameters eam_paras, const rbmd::Real cut_off,
                  const rbmd::Id num_atoms, const rbmd::Id* atoms_type,
                  const rbmd::Id* atoms_id, const rbmd::Id* start_id,
                  const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
                  const Real7* rhor_spline, const Real7* z2r_spline,
                  const rbmd::Real* px,
                  const rbmd::Real* py, const rbmd::Real* pz,
                  rbmd::Real* eam_fp, rbmd::Real* fx, rbmd::Real* fy,
                  rbmd::Real* fz, rbmd::Real* energy_pair);
};

template <>
struct ComputeFpDirect<device::DEVICE_GPU> {
  void operator()(Box box, EAMParameters eam_paras, rbmd::Real cut_off,
                  rbmd::Id num_atoms, EAMDirectTraversalData traversal,
                  const Real7* rhor_spline, const Real7* frho_spline,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, rbmd::Real* eam_fp,
                  rbmd::Real* energy_embedding);
};

template <>
struct ComputeEAMForceRBLDirect<device::DEVICE_GPU> {
  void operator()(Box box, EAMParameters eam_paras, rbmd::Real r_core,
                  rbmd::Real cut_off, rbmd::Id num_atoms,
                  rbmd::Id neighbor_sample_num,
                  rbmd::Id selection_frequency,
                  EAMDirectTraversalData traversal,
                  const Real7* rhor_spline, const Real7* z2r_spline,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* eam_fp,
                  rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz);
};

template <>
struct ComputeEAMEnergyDirect<device::DEVICE_GPU> {
  void operator()(Box box, EAMParameters eam_paras, rbmd::Real cut_off,
                  rbmd::Id num_atoms, EAMDirectTraversalData traversal,
                  const Real7* rhor_spline, const Real7* z2r_spline,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* eam_fp,
                  rbmd::Real* energy_pair);
};


template <>
struct ComputeRhoRBL<device::DEVICE_GPU>
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real rs,
        const rbmd::Real rc,
        const rbmd::Id num_atoms,
        const rbmd::Id neighbor_sample_num,
        const rbmd::Id pice_num,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const rbmd::Id* id_random_neighbor,
        const rbmd::Id* random_neighbor_num,
        const Real7*  rhor_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        rbmd::Real* eam_rho);
};



template <>
struct ComputeFpRBL<device::DEVICE_GPU>
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real rs,
        const rbmd::Real rc,
        const rbmd::Id num_atoms,
        const rbmd::Id neighbor_sample_num,
        const rbmd::Id pice_num,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const rbmd::Id* id_random_neighbor,
        const rbmd::Id* random_neighbor_num,
        const Real7*  rhor_spline,
        const Real7* frho_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        rbmd::Real* eam_fp);
};

template <>
struct ComputeEAMForceRBL<device::DEVICE_GPU>
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real rs,
        const rbmd::Real rc,
        const rbmd::Id num_atoms,
        const rbmd::Id neighbor_sample_num,
        const rbmd::Id pice_num,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const rbmd::Id* id_random_neighbor,
        const rbmd::Id* random_neighbor_num,
        const Real7*  rhor_spline,
        const Real7*  z2r_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        const  rbmd::Real* eam_fp,
        rbmd::Real* fx,
        rbmd::Real* fy,
        rbmd::Real* fz);
};

template <>
struct ComputeEAMEnergy<device::DEVICE_GPU>
{
  void operator()( Box box,
        EAMParameters eam_paras,
        const rbmd::Real cut_off,
        const rbmd::Id num_atoms,
        const rbmd::Id* atoms_type,
        const rbmd::Id* atoms_id,
        const rbmd::Id* start_id,
        const rbmd::Id* end_id,
        const rbmd::Id* id_verletlist,
        const Real7* rhor_spline,
        const Real7* z2r_spline,
        const rbmd::Real* px,
        const rbmd::Real* py,
        const rbmd::Real* pz,
        const rbmd::Real* eam_fp,
        rbmd::Real* energy_pair);
};

}
