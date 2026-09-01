//#include <hip/hip_runtime.h>

#include "../common/rbmd_define.h"
#include "eam_op.h"
#include "model/box.h"

namespace op {

//device
__device__ void  EAMRho(
    rbmd::Real rsq, EAMParameters eam_paras,
    const Real7* rhor_spline,rbmd::Real& local_rho)
{
  rbmd::Real r = SQRT(rsq);
  rbmd::Real p = r * (1.0 / eam_paras.dr) + 1.0;
  rbmd::Id m = static_cast<rbmd::Id>(p);
  m = MIN(m, eam_paras.nr - 1);
  p -= m;
  p = MIN(p, 1.0);

  Real7 coeff = rhor_spline[m];
  //printf("coeff[6]:  %f\n",coeff[6]);
  local_rho = ((coeff[3] * p + coeff[4]) * p + coeff[5]) * p + coeff[6];
}

__device__ void  EAMFp(
    EAMParameters eam_paras,const Real7* frho_spline,
    const rbmd::Real eam_rho,rbmd::Real& eam_fp ,rbmd::Real& local_phi)
{
  rbmd::Real p = eam_rho * (1.0 / eam_paras.drho) + 1.0;
  rbmd::Id m = static_cast<rbmd::Id>(p);
  m = MAX(1, MIN(m, eam_paras.nrho - 1));
  p -= m;
  p = MIN(p, 1.0);

  //
  Real7 coeff = frho_spline[m];  //
  eam_fp = (coeff[0] * p + coeff[1]) * p + coeff[2];
  //printf("fp:  %f\n",eam_fp[tid1]);

  // phi
  rbmd::Real scale_type = 1.0;
  local_phi = ((coeff[3] * p +  coeff[4]) * p + coeff[5]) * p + coeff[6];
  if (eam_rho > eam_paras.rhomax) {
    local_phi +=  eam_fp * (eam_rho- eam_paras.rhomax);
  }
  local_phi *= scale_type;
}

__device__ void EAMForce(rbmd::Real rsq, EAMParameters eam_paras,
    const Real7* rhor_spline,const Real7* z2r_spline, const rbmd::Real fp_i,
    const rbmd::Real fp_j, rbmd::Real& force_pair,rbmd::Real& phi)
{
  auto r = SQRT(rsq);
  auto rdr = 1 / eam_paras.dr;
  auto p = r * rdr + 1.0;
  auto m = static_cast<int>(p);
  //Id m = p;
  m = MIN(m, eam_paras.nr - 1);
  p -= m;
  p = MIN(p, 1.0);

  // rhoip = derivative of (density at atom j due to atom i)
  // rhojp = derivative of (density at atom i due to atom j)
  // phi = pair potential energy
  // phip = phi'
  // z2 = phi * r
  // z2p = (phi * r)' = (phi' r) + phi
  // psip needs both fp[i] and fp[j] terms since r_ij appears in two
  //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
  //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip
  // scale factor can be applied by thermodynamic integration

  Real7 coeffi = rhor_spline[m];
  auto rhoip = (coeffi[0] * p + coeffi[1]) * p + coeffi[2];

  auto coeffj = rhor_spline[m];
  auto rhojp = (coeffj[0] * p + coeffj[1]) * p + coeffj[2];

  auto coeff = z2r_spline[m];
  auto z2p = (coeff[0] * p + coeff[1]) * p + coeff[2];
  auto z2 = ((coeff[3] * p + coeff[4]) * p + coeff[5]) * p + coeff[6];

  auto recip = 1.0 / r;
  phi = z2 * recip;                 //pair potential energy
  auto phip = z2p * recip - phi * recip; //pair force

  auto psip = fp_i * rhojp + fp_j * rhoip + phip;
  force_pair = -psip * recip;

  // sum_fx += force_pair * x12;
  // sum_fy += force_pair * y12;
  // sum_fz += force_pair * z12;
}

__device__ rbmd::Id WrapEAMDirectCellIndex(long long value, rbmd::Id width) {
  long long wrapped = value % static_cast<long long>(width);
  if (wrapped < 0) {
    wrapped += width;
  }
  return static_cast<rbmd::Id>(wrapped);
}

__device__ rbmd::Id MapEAMDirectDimension(rbmd::Id cell, long long offset,
                                          rbmd::Id dimension_width,
                                          rbmd::Id native_start,
                                          rbmd::Id native_width,
                                          bool covers_whole_domain) {
  if (covers_whole_domain) {
    const long long native_idx =
        static_cast<long long>(cell) - native_start;
    return WrapEAMDirectCellIndex(native_idx + offset, native_width) +
           native_start;
  }
  return WrapEAMDirectCellIndex(static_cast<long long>(cell) + offset,
                                dimension_width);
}

__device__ rbmd::Id EAMDirectNeighborCell(
    const LinkedCellDeviceDataPtr* linked_cell, rbmd::Id cell_idx,
    rbmd::Id stencil_idx, rbmd::Id cell_count_within_cutoff) {
  const rbmd::Id dim_x = linked_cell->_d_per_dimension_cells[0];
  const rbmd::Id dim_y = linked_cell->_d_per_dimension_cells[1];
  const rbmd::Id dim_z = linked_cell->_d_per_dimension_cells[2];
  const rbmd::Id width = 2 * cell_count_within_cutoff + 1;
  const long long offset_x = stencil_idx % width - cell_count_within_cutoff;
  const long long offset_y =
      (stencil_idx / width) % width - cell_count_within_cutoff;
  const long long offset_z =
      stencil_idx / (width * width) - cell_count_within_cutoff;

  const rbmd::Id cell_z = cell_idx / (dim_x * dim_y);
  const rbmd::Id cell_y = (cell_idx - cell_z * dim_x * dim_y) / dim_x;
  const rbmd::Id cell_x = cell_idx - dim_x * (cell_y + dim_y * cell_z);

  const rbmd::Id neighbor_x = MapEAMDirectDimension(
      cell_x, offset_x, dim_x, linked_cell->_d_nativate_cells_start[0],
      linked_cell->_d_nativate_cells_dims[0],
      linked_cell->_d_covers_whole_domain[0]);
  const rbmd::Id neighbor_y = MapEAMDirectDimension(
      cell_y, offset_y, dim_y, linked_cell->_d_nativate_cells_start[1],
      linked_cell->_d_nativate_cells_dims[1],
      linked_cell->_d_covers_whole_domain[1]);
  const rbmd::Id neighbor_z = MapEAMDirectDimension(
      cell_z, offset_z, dim_z, linked_cell->_d_nativate_cells_start[2],
      linked_cell->_d_nativate_cells_dims[2],
      linked_cell->_d_covers_whole_domain[2]);
  return (neighbor_z * dim_y + neighbor_y) * dim_x + neighbor_x;
}

__global__ void FpDirect(
    Box box, EAMParameters eam_paras, rbmd::Real cut_off, rbmd::Id num_atoms,
    EAMDirectTraversalData traversal, const Real7* rhor_spline,
    const Real7* frho_spline, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* eam_fp,
    rbmd::Real* energy_embedding) {
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage storage;
  rbmd::Real embedding = 0;
  const rbmd::Id atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < num_atoms) {
    rbmd::Real rho = 0;
    const rbmd::Real cutoff_sq = cut_off * cut_off;
    const rbmd::Id cell = traversal.per_atom_cell_id[atom];
    for (rbmd::Id stencil = 0; stencil < traversal.neighbor_cell_num;
         ++stencil) {
      const rbmd::Id neighbor_cell = EAMDirectNeighborCell(
          traversal.linked_cell, cell, stencil,
          traversal.cell_count_within_cutoff);
      const rbmd::Id begin = traversal.cell_atom_start[neighbor_cell];
      const rbmd::Id end = traversal.cell_atom_end[neighbor_cell];
      for (rbmd::Id pos = begin; pos < end; ++pos) {
        const rbmd::Id neighbor = traversal.sorted_atom_indices[pos];
        if (neighbor == atom) {
          continue;
        }
        rbmd::Real dx = px[atom] - px[neighbor];
        rbmd::Real dy = py[atom] - py[neighbor];
        rbmd::Real dz = pz[atom] - pz[neighbor];
        MinImageDistance(box, dx, dy, dz);
        const rbmd::Real rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < cutoff_sq) {
          rbmd::Real local_rho = 0;
          EAMRho(rsq, eam_paras, rhor_spline, local_rho);
          rho += local_rho;
        }
      }
    }
    EAMFp(eam_paras, frho_spline, rho, eam_fp[atom], embedding);
  }
  const rbmd::Real block_sum =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(storage).Sum(embedding);
  if (threadIdx.x == 0) {
    atomicAdd(energy_embedding, block_sum);
  }
}

__global__ void EAMForceRBLDirect(
    Box box, EAMParameters eam_paras, rbmd::Real r_core,
    rbmd::Real cut_off, rbmd::Id num_atoms, rbmd::Id neighbor_sample_num,
    rbmd::Id selection_frequency, EAMDirectTraversalData traversal,
    const Real7* rhor_spline, const Real7* z2r_spline,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* eam_fp, rbmd::Real* fx, rbmd::Real* fy,
    rbmd::Real* fz) {
  const rbmd::Id atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= num_atoms) {
    return;
  }

  rbmd::Real core_fx = 0;
  rbmd::Real core_fy = 0;
  rbmd::Real core_fz = 0;
  rbmd::Real shell_fx = 0;
  rbmd::Real shell_fy = 0;
  rbmd::Real shell_fz = 0;
  rbmd::Id shell_index = 0;
  rbmd::Id sampled = 0;
  const rbmd::Real core_sq = r_core * r_core;
  const rbmd::Real cutoff_sq = cut_off * cut_off;
  const rbmd::Id cell = traversal.per_atom_cell_id[atom];

  for (rbmd::Id stencil = 0; stencil < traversal.neighbor_cell_num;
       ++stencil) {
    const rbmd::Id neighbor_cell = EAMDirectNeighborCell(
        traversal.linked_cell, cell, stencil,
        traversal.cell_count_within_cutoff);
    const rbmd::Id begin = traversal.cell_atom_start[neighbor_cell];
    const rbmd::Id end = traversal.cell_atom_end[neighbor_cell];
    for (rbmd::Id pos = begin; pos < end; ++pos) {
      const rbmd::Id neighbor = traversal.sorted_atom_indices[pos];
      if (neighbor == atom) {
        continue;
      }
      rbmd::Real dx = px[atom] - px[neighbor];
      rbmd::Real dy = py[atom] - py[neighbor];
      rbmd::Real dz = pz[atom] - pz[neighbor];
      MinImageDistance(box, dx, dy, dz);
      const rbmd::Real rsq = dx * dx + dy * dy + dz * dz;
      if (rsq < core_sq) {
        rbmd::Real pair_force = 0;
        rbmd::Real phi = 0;
        EAMForce(rsq, eam_paras, rhor_spline, z2r_spline, eam_fp[atom],
                 eam_fp[neighbor], pair_force, phi);
        core_fx += pair_force * dx;
        core_fy += pair_force * dy;
        core_fz += pair_force * dz;
      } else if (rsq < cutoff_sq) {
        if ((shell_index % selection_frequency) == 0 &&
            sampled < neighbor_sample_num) {
          rbmd::Real pair_force = 0;
          rbmd::Real phi = 0;
          EAMForce(rsq, eam_paras, rhor_spline, z2r_spline, eam_fp[atom],
                   eam_fp[neighbor], pair_force, phi);
          shell_fx += pair_force * dx;
          shell_fy += pair_force * dy;
          shell_fz += pair_force * dz;
          ++sampled;
        }
        ++shell_index;
      }
    }
  }
  fx[atom] = core_fx + selection_frequency * shell_fx;
  fy[atom] = core_fy + selection_frequency * shell_fy;
  fz[atom] = core_fz + selection_frequency * shell_fz;
}

__global__ void EAMEnergyDirect(
    Box box, EAMParameters eam_paras, rbmd::Real cut_off,
    rbmd::Id num_atoms, EAMDirectTraversalData traversal,
    const Real7* rhor_spline, const Real7* z2r_spline,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* eam_fp, rbmd::Real* energy_pair) {
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage storage;
  rbmd::Real pair_energy = 0;
  const rbmd::Id atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < num_atoms) {
    const rbmd::Real cutoff_sq = cut_off * cut_off;
    const rbmd::Id cell = traversal.per_atom_cell_id[atom];
    for (rbmd::Id stencil = 0; stencil < traversal.neighbor_cell_num;
         ++stencil) {
      const rbmd::Id neighbor_cell = EAMDirectNeighborCell(
          traversal.linked_cell, cell, stencil,
          traversal.cell_count_within_cutoff);
      const rbmd::Id begin = traversal.cell_atom_start[neighbor_cell];
      const rbmd::Id end = traversal.cell_atom_end[neighbor_cell];
      for (rbmd::Id pos = begin; pos < end; ++pos) {
        const rbmd::Id neighbor = traversal.sorted_atom_indices[pos];
        if (neighbor == atom) {
          continue;
        }
        rbmd::Real dx = px[atom] - px[neighbor];
        rbmd::Real dy = py[atom] - py[neighbor];
        rbmd::Real dz = pz[atom] - pz[neighbor];
        MinImageDistance(box, dx, dy, dz);
        const rbmd::Real rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < cutoff_sq) {
          rbmd::Real pair_force = 0;
          rbmd::Real phi = 0;
          EAMForce(rsq, eam_paras, rhor_spline, z2r_spline, eam_fp[atom],
                   eam_fp[neighbor], pair_force, phi);
          pair_energy += phi;
        }
      }
    }
  }
  const rbmd::Real block_sum =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(storage).Sum(0.5 * pair_energy);
  if (threadIdx.x == 0) {
    atomicAdd(energy_pair, block_sum);
  }
}


//Verlet :: EAM Rho  to fp
__global__ void FpVerlet(
    Box box, EAMParameters eam_paras ,const rbmd::Real rc,
    const rbmd::Id num_atoms, const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const Real7* rhor_spline,const Real7* frho_spline,
    const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz,rbmd::Real* eam_fp,rbmd::Real*  energy_embedding) {
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
  temp_storage;

  rbmd::Real  sum_rho= 0;
  rbmd::Real  sum_phi= 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];

    //
    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      rbmd::Id typej = atoms_type[tid2];

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rc_2 = rc * rc;
      rbmd::Real local_rho = 0;

      //Rho
      if (rsq < rc_2 )
      {
        EAMRho(rsq,eam_paras,rhor_spline,local_rho);
      }
      sum_rho  += local_rho;
    }

    //Fp
    rbmd::Real  local_eam_fp= 0;
    rbmd::Real local_phi= 0;
    EAMFp(eam_paras,frho_spline,sum_rho,local_eam_fp,local_phi);

    eam_fp[tid1] = local_eam_fp ;
    sum_phi = local_phi;

  }

  rbmd::Real block_sum =
BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
    .Sum(sum_phi);

  if (threadIdx.x == 0) {
    atomicAdd(energy_embedding, block_sum);
  }
}

//Verlet :: EAMForce
__global__ void EAMForceVerlet(
  Box box, EAMParameters eam_paras ,const rbmd::Real rc,const rbmd::Id num_atoms,
  const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
  const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
  const Real7* rhor_spline,const Real7* z2r_spline,
  const rbmd::Real* px, const rbmd::Real* py,const rbmd::Real* pz,
  const rbmd::Real* eam_fp,rbmd::Real* fx,rbmd::Real* fy, rbmd::Real* fz,
  rbmd::Real* energy_pair)
{
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
  temp_storage;
  rbmd::Real local_phi =0.0;

  //init
  rbmd::Real sum_fx = 0;
  rbmd::Real sum_fy = 0;
  rbmd::Real sum_fz = 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real fp_i = eam_fp[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];

    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      rbmd::Id typej = atoms_type[tid2];
      rbmd::Real fp_j = eam_fp[tid2];

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];

      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      // rbmd::Real x12 = x2 - x1;
      // rbmd::Real y12 = y2 - y1;
      // rbmd::Real z12 = z2 - z1;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real cutsq = rc * rc;
      rbmd::Real force_pair = 0.0;
      rbmd::Real phi = 0.0;
      if (rsq < cutsq)
      {
        EAMForce(rsq,eam_paras,rhor_spline,z2r_spline,fp_i,fp_j,
          force_pair,phi);

      }
      sum_fx += force_pair * x12;
      sum_fy += force_pair * y12;
      sum_fz += force_pair * z12;
      local_phi += phi;
    }
    //compute f
    fx[tid1]  = sum_fx;
    fy[tid1]  = sum_fy;
    fz[tid1]  = sum_fz;
  }

  rbmd::Real block_sum =
  BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
      .Sum(0.5 * local_phi);

  if (threadIdx.x == 0) {
    atomicAdd(energy_pair, block_sum);
  }

}


//RBL :: EAMRhoRBL
__global__ void RhoRBL(
    Box box, EAMParameters eam_paras ,const rbmd::Real rs,const rbmd::Real rc,
    const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
    const rbmd::Id pice_num,const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const rbmd::Id* id_random_neighbor,const rbmd::Id* random_neighbor_num,
    const Real7* rhor_spline,const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz,rbmd::Real* eam_rho) {

  rbmd::Real sum_rho_rs = 0;
  rbmd::Real sum_rho_rcs = 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];

    // rs
    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      rbmd::Id typej = atoms_type[tid2];

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rs_2 = rs * rs;
      rbmd::Real local_rho_rs = 0;

      if (rsq < rs_2 )
      {
        EAMRho(rsq,eam_paras,rhor_spline,local_rho_rs);
      }
      sum_rho_rs += local_rho_rs;
    }

    //rcs
    rbmd::Id real_random_num = random_neighbor_num[tid1];
    for (rbmd::Id jj = 0; jj < real_random_num; ++jj)
    {
      const unsigned long random_neighbor_index =
          static_cast<unsigned long>(jj) *
              static_cast<unsigned long>(num_atoms) +
          static_cast<unsigned long>(tid1);
      rbmd::Id tid2 = id_random_neighbor[random_neighbor_index];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      rbmd::Id atom_id2 = atoms_id[tid2];
      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rs_2 = rs* rs;
      rbmd::Real rc_2 = rc* rc;
      rbmd::Real local_rho_rcs = 0;
      if (rsq < rc_2 && rsq > rs_2) {
        EAMRho(rsq,eam_paras,rhor_spline,local_rho_rcs);
      }
      sum_rho_rcs += local_rho_rcs;
    }

    eam_rho[tid1] = sum_rho_rs + pice_num * sum_rho_rcs;
  }

}


__global__ void FpRBL(
    Box box, EAMParameters eam_paras ,const rbmd::Real rs,const rbmd::Real rc,
    const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
    const rbmd::Id pice_num,const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const rbmd::Id* id_random_neighbor,const rbmd::Id* random_neighbor_num,
    const Real7* rhor_spline,const Real7* frho_spline,
    const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz,rbmd::Real* eam_fp) {

  rbmd::Real sum_rho_rs = 0;
  rbmd::Real sum_rho_crs = 0;
  rbmd::Real  eam_fp_rs= 0;
  rbmd::Real  eam_fp_rcs= 0;

  rbmd::Real  phi_rs= 0;
  rbmd::Real  phi_rcs= 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];

    // rs
    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      rbmd::Id typej = atoms_type[tid2];

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rs_2 = rs * rs;
      rbmd::Real local_rho_rs = 0;

      if (rsq < rs_2 )
      {
        EAMRho(rsq,eam_paras,rhor_spline,local_rho_rs);
      }
      sum_rho_rs += local_rho_rs;
    }

    rbmd::Real local_phi_rs = 0;
    EAMFp(eam_paras,frho_spline,sum_rho_rs,eam_fp_rs,local_phi_rs);
    phi_rs += local_phi_rs;


    //rcs
    rbmd::Id real_random_num = random_neighbor_num[tid1];
    for (rbmd::Id jj = 0; jj < real_random_num; ++jj)
    {
      const unsigned long random_neighbor_index =
          static_cast<unsigned long>(jj) *
              static_cast<unsigned long>(num_atoms) +
          static_cast<unsigned long>(tid1);
      rbmd::Id tid2 = id_random_neighbor[random_neighbor_index];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      rbmd::Id atom_id2 = atoms_id[tid2];
      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rs_2 = rs* rs;
      rbmd::Real rc_2 = rc* rc;
      rbmd::Real local_rho_rcs = 0;

      if (rsq < rc_2 && rsq > rs_2) {
        EAMRho(rsq,eam_paras,rhor_spline,local_rho_rcs);
      }
     sum_rho_crs +=  local_rho_rcs;
    }

    rbmd::Real local_phi_rcs = 0;
    EAMFp(eam_paras,frho_spline,sum_rho_crs,eam_fp_rcs,local_phi_rcs);
    phi_rcs += local_phi_rcs;

    eam_fp[tid1] = eam_fp_rs + pice_num * eam_fp_rcs;
  }

}

__global__ void EAMForceRBL(
  Box box, EAMParameters eam_paras ,const rbmd::Real rs,const rbmd::Real rc,
  const rbmd::Id num_atoms,const rbmd::Id neighbor_sample_num,
  const rbmd::Id pice_num,
  const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
  const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
  const rbmd::Id* id_random_neighbor,const rbmd::Id* random_neighbor_num,
  const Real7* rhor_spline,const Real7* z2r_spline,
  const rbmd::Real* px, const rbmd::Real* py,const rbmd::Real* pz,
  const rbmd::Real* eam_fp,rbmd::Real* fx,rbmd::Real* fy, rbmd::Real* fz)
{
  rbmd::Real phi_rs =0.0;
  rbmd::Real  phi_rcs=0.0;

  //init
  rbmd::Real sum_fx = 0;
  rbmd::Real sum_fy = 0;
  rbmd::Real sum_fz = 0;
  rbmd::Real sum_fsx = 0;
  rbmd::Real sum_fsy = 0;
  rbmd::Real sum_fsz = 0;
  rbmd::Real sum_fcsx = 0;
  rbmd::Real sum_fcsy = 0;
  rbmd::Real sum_fcsz = 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real fp_i = eam_fp[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];

    //rs
    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      rbmd::Id typej = atoms_type[tid2];
      rbmd::Real fp_j = eam_fp[tid2];

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rs_2 = rs * rs;
      rbmd::Real local_phi_rs =0.0;
      rbmd::Real force_pair = 0.0;
      if (rsq < rs_2)
      {
        EAMForce(rsq,eam_paras,rhor_spline,z2r_spline,fp_i,fp_j,
          force_pair,local_phi_rs);
      }

      sum_fsx += force_pair * x12;
      sum_fsy += force_pair * y12;
      sum_fsz += force_pair * z12;
      phi_rs += local_phi_rs;
    }

    //rcs
    rbmd::Id real_random_num = random_neighbor_num[tid1];
    for (rbmd::Id jj = 0; jj < real_random_num; ++jj) {
      const unsigned long random_neighbor_index =
          static_cast<unsigned long>(jj) *
              static_cast<unsigned long>(num_atoms) +
          static_cast<unsigned long>(tid1);
      rbmd::Id tid2 = id_random_neighbor[random_neighbor_index];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      rbmd::Id atom_id2 = atoms_id[tid2];
      rbmd::Real fp_j = eam_fp[tid2];

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rs_2 = rs* rs;
      rbmd::Real rc_2 = rc* rc;
      rbmd::Real force_pair = 0.0;
      rbmd::Real local_phi_rcs = 0.0;
      if (rsq < rc_2 && rsq > rs_2) {
        EAMForce(rsq,eam_paras,rhor_spline,z2r_spline,fp_i,fp_j,
          force_pair,local_phi_rcs);
      }
      sum_fcsx += force_pair * x12;
      sum_fcsy += force_pair * y12;
      sum_fcsz += force_pair * z12;

      phi_rcs += local_phi_rcs;
    }

    //compute f
    sum_fx = sum_fsx + pice_num* sum_fcsx;
    sum_fy = sum_fsy + pice_num* sum_fcsy;
    sum_fz = sum_fsz + pice_num* sum_fcsz;

    //printf("sum_fx: %f\n",sum_fx);
    fx[tid1]  = sum_fx;
    fy[tid1]  = sum_fy;
    fz[tid1]  = sum_fz;
  }

}


__global__ void EAMForceEnergy(
  Box box, EAMParameters eam_paras ,const rbmd::Real rc,const rbmd::Id num_atoms,
  const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
  const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
  const Real7* rhor_spline,const Real7* z2r_spline,
  const rbmd::Real* px, const rbmd::Real* py,const rbmd::Real* pz,
  const rbmd::Real* eam_fp,  rbmd::Real* energy_pair)
{
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
  temp_storage;

  //init
  rbmd::Real sum_phi =0.0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real fp_i = eam_fp[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];

    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      rbmd::Id typej = atoms_type[tid2];
      rbmd::Real fp_j = eam_fp[tid2];

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];

      rbmd::Real x12 = x1 - x2;
      rbmd::Real y12 = y1 - y2;
      rbmd::Real z12 = z1 - z2;

      // rbmd::Real x12 = x2 - x1;
      // rbmd::Real y12 = y2 - y1;
      // rbmd::Real z12 = z2 - z1;

      MinImageDistance(box, x12, y12, z12);
      rbmd::Real rsq = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real rc_2 = rc * rc;
      rbmd::Real force_pair = 0.0;
      rbmd::Real local_phi = 0.0;
      if (rsq < rc_2)
      {
        EAMForce(rsq,eam_paras,rhor_spline,z2r_spline,fp_i,fp_j,
          force_pair,local_phi);
      }
      sum_phi += local_phi;
    }
  }

  rbmd::Real block_sum =
  BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
      .Sum(0.5 * sum_phi);

  if (threadIdx.x == 0) {
    atomicAdd(energy_pair, block_sum);
  }
}



////////////////////////
// Verlet ::  Fp
void ComputeFpVerlet<device::DEVICE_GPU>::operator()(
Box box, EAMParameters eam_paras ,const rbmd::Real rc,const rbmd::Id num_atoms,
const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
const Real7* rhor_spline,const Real7* frho_spline,
const rbmd::Real* px, const rbmd::Real* py,const rbmd::Real* pz,
rbmd::Real* eam_fp ,rbmd::Real* energy_embedding) {
  unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // 1.  rho  ->  Fp
  CHECK_KERNEL(FpVerlet<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      box, eam_paras, rc, num_atoms, atoms_type, atoms_id, start_id, end_id,
      id_verletlist, rhor_spline, frho_spline, px, py, pz, eam_fp,
      energy_embedding));
}

// Verlet :: EAMForceVerletOp
void ComputeEAMForceVerlet<device::DEVICE_GPU>::operator()(
    Box box, EAMParameters eam_paras, const rbmd::Real rc,
    const rbmd::Id num_atoms, const rbmd::Id* atoms_type,
    const rbmd::Id* atoms_id, const rbmd::Id* start_id, const rbmd::Id* end_id,
    const rbmd::Id* id_verletlist, const Real7* rhor_spline,
    const Real7* z2r_spline, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* eam_fp, rbmd::Real* fx,
    rbmd::Real* fy, rbmd::Real* fz, rbmd::Real* energy_pair) {
  unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(EAMForceVerlet<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>> (
      box, eam_paras, rc, num_atoms, atoms_type, atoms_id, start_id, end_id,
      id_verletlist, rhor_spline, z2r_spline, px, py, pz, eam_fp, fx, fy, fz,
      energy_pair));
}

void ComputeFpDirect<device::DEVICE_GPU>::operator()(
    Box box, EAMParameters eam_paras, rbmd::Real cut_off,
    rbmd::Id num_atoms, EAMDirectTraversalData traversal,
    const Real7* rhor_spline, const Real7* frho_spline,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    rbmd::Real* eam_fp, rbmd::Real* energy_embedding) {
  const unsigned int blocks_per_grid =
      (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(FpDirect<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      box, eam_paras, cut_off, num_atoms, traversal, rhor_spline,
      frho_spline, px, py, pz, eam_fp, energy_embedding));
}

void ComputeEAMForceRBLDirect<device::DEVICE_GPU>::operator()(
    Box box, EAMParameters eam_paras, rbmd::Real r_core,
    rbmd::Real cut_off, rbmd::Id num_atoms, rbmd::Id neighbor_sample_num,
    rbmd::Id selection_frequency, EAMDirectTraversalData traversal,
    const Real7* rhor_spline, const Real7* z2r_spline,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* eam_fp, rbmd::Real* fx, rbmd::Real* fy,
    rbmd::Real* fz) {
  const unsigned int blocks_per_grid =
      (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(EAMForceRBLDirect<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      box, eam_paras, r_core, cut_off, num_atoms, neighbor_sample_num,
      selection_frequency, traversal, rhor_spline, z2r_spline, px, py, pz,
      eam_fp, fx, fy, fz));
}

void ComputeEAMEnergyDirect<device::DEVICE_GPU>::operator()(
    Box box, EAMParameters eam_paras, rbmd::Real cut_off,
    rbmd::Id num_atoms, EAMDirectTraversalData traversal,
    const Real7* rhor_spline, const Real7* z2r_spline,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* eam_fp, rbmd::Real* energy_pair) {
  const unsigned int blocks_per_grid =
      (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(EAMEnergyDirect<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      box, eam_paras, cut_off, num_atoms, traversal, rhor_spline,
      z2r_spline, px, py, pz, eam_fp, energy_pair));
}

//RBL::RhoRBL
void ComputeRhoRBL<device::DEVICE_GPU>::operator()(
    Box box, EAMParameters eam_paras ,const rbmd::Real rs,const rbmd::Real rc,
    const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
    const rbmd::Id pice_num,const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const rbmd::Id* id_random_neighbor,const rbmd::Id* random_neighbor_num,
    const Real7* rhor_spline,const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz,rbmd::Real* eam_rho) {
  unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(RhoRBL<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>> (
          box, eam_paras,rs,rc, num_atoms,neighbor_sample_num,pice_num,atoms_type,
          atoms_id,  start_id, end_id,id_verletlist,id_random_neighbor,
          random_neighbor_num,rhor_spline,px, py, pz,eam_rho));

}

//RBL::Fp
void ComputeFpRBL<device::DEVICE_GPU>::operator()(
    Box box, EAMParameters eam_paras ,const rbmd::Real rs,const rbmd::Real rc,
    const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
    const rbmd::Id pice_num,const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const rbmd::Id* id_random_neighbor,const rbmd::Id* random_neighbor_num,
    const Real7* rhor_spline,const Real7* frho_spline,
    const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz,rbmd::Real* eam_fp) {
  unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(FpRBL<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>> (
          box, eam_paras,rs,rc, num_atoms,neighbor_sample_num,pice_num,atoms_type,
          atoms_id,  start_id, end_id,id_verletlist,id_random_neighbor,
          random_neighbor_num,rhor_spline,frho_spline,px, py, pz,eam_fp));
}




//RBL:: EAMForce
void ComputeEAMForceRBL<device::DEVICE_GPU>::operator()(
    Box box, EAMParameters eam_paras ,const rbmd::Real rs,const rbmd::Real rc,
    const rbmd::Id num_atoms,const rbmd::Id neighbor_sample_num,
    const rbmd::Id pice_num,
    const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const rbmd::Id* id_random_neighbor,const rbmd::Id* random_neighbor_num,
    const Real7* rhor_spline,const Real7* z2r_spline,
    const rbmd::Real* px, const rbmd::Real* py,const rbmd::Real* pz,
    const rbmd::Real* eam_fp,rbmd::Real* fx,rbmd::Real* fy, rbmd::Real* fz) {
  unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(EAMForceRBL<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>> (
          box, eam_paras,rs,rc, num_atoms,neighbor_sample_num,pice_num,
          atoms_type, atoms_id,
          start_id, end_id,id_verletlist,
          id_random_neighbor,random_neighbor_num,
          rhor_spline,z2r_spline,px, py, pz,eam_fp,
          fx,fy,fz));
}

void ComputeEAMEnergy<device::DEVICE_GPU>::operator()(
     Box box, EAMParameters eam_paras ,const rbmd::Real rc,const rbmd::Id num_atoms,
     const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const Real7* rhor_spline,const Real7* z2r_spline,
    const rbmd::Real* px, const rbmd::Real* py,const rbmd::Real* pz,
    const rbmd::Real* eam_fp,rbmd::Real* energy_pair) {
  unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(EAMForceEnergy<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>> (
          box, eam_paras,rc, num_atoms,atoms_type, atoms_id,  start_id, end_id,
          id_verletlist,rhor_spline,z2r_spline,px, py, pz,eam_fp,energy_pair));

}

}
