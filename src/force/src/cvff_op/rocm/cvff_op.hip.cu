//#include <hip/hip_runtime.h>

#include "../common/rbmd_define.h"
#include "cvff_op.h"
#include "model/box.h"
#include "../lj_cut_coul_kspace_op/rocm/lj_cut_coul_kspace_op.hip.cu"

const rbmd::Real SMALL = 0.001;
const rbmd::Real SMALLER = 0.00001;
namespace op{

__global__ void reduce_virial_kernel(
    const rbmd::Id num_atoms,const rbmd::Id pitch,
    const rbmd::Real* d_flat_virial_atom,rbmd::Real* d_virial)
{
  extern __shared__ rbmd::Real sdata[];  //

  //
  int j = blockIdx.x;
  int tid = threadIdx.x;
  rbmd::Real sum = 0.0;

  for (int atom = tid; atom < num_atoms; atom += blockDim.x) {
    sum += d_flat_virial_atom[j * pitch + atom];
  }

  //
  sdata[tid] = sum;
  __syncthreads();

  //
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  //
  if (tid == 0) {
    d_virial[j] = sdata[0];
  }
}

inline __device__ void LoadSpecialCounts(unsigned int tid1,
                                         const int* nspecial,
                                         int maxspecial,
                                         int& n12,
                                         int& n13,
                                         int& n14) {
  n12 = 0;
  n13 = 0;
  n14 = 0;
  if (!nspecial || maxspecial <= 0) {
    return;
  }

  const int base = static_cast<int>(tid1) * 3;
  n12 = nspecial[base + 0];
  n13 = nspecial[base + 1];
  n14 = nspecial[base + 2];

  if (n12 < 0) n12 = 0;
  if (n13 < 0) n13 = 0;
  if (n14 < 0) n14 = 0;
  if (n12 > n13) n13 = n12;
  if (n13 > n14) n14 = n13;
  if (n14 > maxspecial) n14 = maxspecial;
  if (n13 > n14) n13 = n14;
  if (n12 > n13) n12 = n13;
}

inline __device__ rbmd::Real LookupSpecialWeight(unsigned int tid1,
                                                 rbmd::Id gid2,
                                                 const int* nspecial,
                                                 const rbmd::Id* special,
                                                 int maxspecial,
                                                 rbmd::Real w12,
                                                 rbmd::Real w13,
                                                 rbmd::Real w14) {
  int n12 = 0;
  int n13 = 0;
  int n14 = 0;
  LoadSpecialCounts(tid1, nspecial, maxspecial, n12, n13, n14);
  if (!special || maxspecial <= 0 || n14 <= 0) {
    return rbmd::Real(1.0);
  }

  const int limit = (n14 < maxspecial) ? n14 : maxspecial;
  const rbmd::Id* row =
      special + static_cast<long long>(tid1) * static_cast<long long>(maxspecial);
  for (int k = 0; k < limit; ++k) {
    if (row[k] == gid2) {
      if (k < n12) {
        return w12;
      }
      if (k < n13) {
        return w13;
      }
      return w14;
    }
  }
  return rbmd::Real(1.0);
}

__global__ void ComputeSpecialLJCutCoulForceUserKernel(
  Box box, const rbmd::Real cut_off, const rbmd::Id num_atoms,const  rbmd::Real qqr2e,
   const rbmd::Real rbsog_sigma,const rbmd::Real rbsog_b, const rbmd::Id rbsog_mmax,
   const rbmd::Real rbsog_w0,const rbmd::Real* taylor_coeff,
   const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
   const rbmd::Real* sigma, const rbmd::Real* eps,
   const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
   const int* nspecial, const rbmd::Id* special, int maxspecial,
   rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
   const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
   rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
   rbmd::Real* flat_virial, rbmd::Real* total_evdwl, rbmd::Real* total_ecoul)
{
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage temp_storage_elj;
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage temp_storage_ecoul;
    rbmd::Real sum_fx = 0;
    rbmd::Real sum_fy = 0;
    rbmd::Real sum_fz = 0;
    rbmd::Real sum_elj = 0;
    rbmd::Real sum_ecoul = 0;
    //virial init
    rbmd::Real sum_virial[6];
    for (int i = 0; i < 6; ++i)
    {
      sum_virial[i] = 0.0;
    }

    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_atoms) {
      const rbmd::Id gid1 = atoms_id[tid1];
      rbmd::Id typei = atoms_type[tid1];
      rbmd::Real eps_i = eps[typei];
      rbmd::Real sigma_i = sigma[typei];
      rbmd::Real charge_i = charge[tid1];
      rbmd::Real x1 = px[tid1];
      rbmd::Real y1 = py[tid1];
      rbmd::Real z1 = pz[tid1];

      for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
        rbmd::Id tid2 = id_verletlist[j];
        if (tid2 == static_cast<rbmd::Id>(tid1)) {
          continue;
        }
        const rbmd::Id gid2 = atoms_id[tid2];
        rbmd::Id typej = atoms_type[tid2];
        rbmd::Real eps_j = eps[typej];
        rbmd::Real sigma_j = sigma[typej];
        rbmd::Real charge_j = charge[tid2];
        // mix
        rbmd::Real eps_ij = SQRT(eps_i * eps_j);
        rbmd::Real sigma_ij = (sigma_i + sigma_j) / 2;
        rbmd::Real x2 = px[tid2];
        rbmd::Real y2 = py[tid2];
        rbmd::Real z2 = pz[tid2];
        rbmd::Real x12 = x2 - x1;
        rbmd::Real y12 = y2 - y1;
        rbmd::Real z12 = z2 - z1;
        // rbmd::Real x12 = x1 - x2;
        // rbmd::Real y12 = y1 - y2;
        // rbmd::Real z12 = z1 - z2;
        MinImageDistance_while(box, x12, y12, z12);


        rbmd::Real force_lj, force_coul, force_pair,force_coul_factor;
        rbmd::Real energy_lj, energy_coul,energy_coul_factor;
          // lj cut
        lj126(cut_off, x12, y12, z12, eps_ij, sigma_ij,
          force_lj, energy_lj);

        // --- . Apply Special Weights ---
        const rbmd::Real weight =
            LookupSpecialWeight(tid1, gid2, nspecial, special, maxspecial, w12,
                                w13, w14);

          // --- . Coul Calculation (RBSOG User) ---
        rbmd::Real f_coul_full, e_coul_full, f_coul_short, e_coul_short;
        CoulCutForceUser(cut_off, x12, y12,z12, qqr2e, charge_i, charge_j,
                           taylor_coeff[0], taylor_coeff[1],
                           taylor_coeff[2],taylor_coeff[3],
                           taylor_coeff[4], taylor_coeff[5],
                           rbsog_sigma, rbsog_b, rbsog_mmax, rbsog_w0,
                           f_coul_full, e_coul_full,
                           f_coul_short, e_coul_short);



        // Formula: Scaled_Short = Short - (1 - weight) * Full
        f_coul_short = f_coul_short- (1.0 - weight) * f_coul_full;
        force_pair = weight * force_lj  + f_coul_short;
        // printf("force_pair:  %f\n",force_pair);

         sum_fx += x12 * force_pair;
         sum_fy += y12 * force_pair;
         sum_fz += z12 * force_pair;

         sum_elj += weight * energy_lj;
         sum_ecoul += (e_coul_short - (1.0 - weight) * e_coul_full);

         // --- . Virial ---
        //rbmd::Real local_virial[6];
        rbmd::Real local_virial_xx,local_virial_yy,local_virial_zz,
          local_virial_xy,local_virial_xz,local_virial_yz;
        ComputeVirial(x12, y12, z12,force_pair,
          local_virial_xx,local_virial_yy,local_virial_zz,
          local_virial_xy,local_virial_xz,local_virial_yz);

        //
        sum_virial[0] +=local_virial_xx;
        sum_virial[1] +=local_virial_yy;
        sum_virial[2] +=local_virial_zz;
        sum_virial[3] +=local_virial_xy;
        sum_virial[4] +=local_virial_xz;
        sum_virial[5] +=local_virial_yz;

      }

      fx[tid1] = sum_fx;
      fy[tid1] = sum_fy;
      fz[tid1] = sum_fz;
      //
      for(int i =0;i<6;++i) {
        flat_virial[  i * num_atoms + tid1] = sum_virial[i];
      }
    }

    rbmd::Real b_sum_elj = BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_elj).Sum(sum_elj);
    rbmd::Real b_sum_ecoul = BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_ecoul).Sum(sum_ecoul);
    if (threadIdx.x == 0) {
        atomicAdd(total_evdwl, b_sum_elj);
        atomicAdd(total_ecoul, b_sum_ecoul);
    }
}

//verlet-list : SpecialLJCutCoul
inline __device__ void LJForceEnergyFromTypePair(
    rbmd::Real distance_2, rbmd::Real cutoff_2, rbmd::Real lj_force12,
    rbmd::Real lj_force6, rbmd::Real& force, rbmd::Real& energy) {
  if (distance_2 >= cutoff_2) {
    force = rbmd::Real(0);
    energy = rbmd::Real(0);
    return;
  }
  const rbmd::Real r2inv = rbmd::Real(1) / distance_2;
  const rbmd::Real r6inv = r2inv * r2inv * r2inv;
  force = -r2inv * r6inv * (lj_force12 * r6inv - lj_force6);
  energy = (lj_force12 * rbmd::Real(1.0 / 24.0)) * r6inv * r6inv -
           (lj_force6 * rbmd::Real(1.0 / 12.0)) * r6inv;
}

__global__ void ComputeSpecialLJCutCoulForce(
     Box box, const rbmd::Real cut_off,
    const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
    const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id lj_num_types, const rbmd::Real* lj_force12,
    const rbmd::Real* lj_force6, const rbmd::Id* start_id,
    const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const int* nspecial, const rbmd::Id* special, int maxspecial,
    rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
    const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
    rbmd::Real* flat_virial,rbmd::Real* total_evdwl, rbmd::Real* total_ecoul) {
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
      temp_storage_elj;
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
      temp_storage_ecoul;

  rbmd::Real sum_fx = 0;
  rbmd::Real sum_fy = 0;
  rbmd::Real sum_fz = 0;
  rbmd::Real sum_elj = 0;
  rbmd::Real sum_ecoul = 0;
  //virial init
  rbmd::Real sum_virial[6];
  for (int i = 0; i < 6; ++i)
  {
    sum_virial[i] = 0.0;
  }

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    const rbmd::Id gid1 = atoms_id[tid1];
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real charge_i = charge[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];

    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      const rbmd::Id gid2 = atoms_id[tid2];
      rbmd::Id typej = atoms_type[tid2];
      rbmd::Real charge_j = charge[tid2];
      const rbmd::Id pair_index = typei * lj_num_types + typej;
      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real x12 = x2 - x1;
      rbmd::Real y12 = y2 - y1;
      rbmd::Real z12 = z2 - z1;
      // rbmd::Real x12 = x1 - x2;
      // rbmd::Real y12 = y1 - y2;
      // rbmd::Real z12 = z1 - z2;
      MinImageDistance_while(box, x12, y12, z12);

      rbmd::Real force_lj, force_coul, force_pair,force_coul_factor;
      rbmd::Real energy_lj, energy_coul,energy_coul_factor;
      // lj cut
      const rbmd::Real distance_2 = x12 * x12 + y12 * y12 + z12 * z12;
      LJForceEnergyFromTypePair(distance_2, cut_off * cut_off,
                                lj_force12[pair_index],
                                lj_force6[pair_index], force_lj, energy_lj);

      // special lj  weight
      const rbmd::Real weight =
          LookupSpecialWeight(tid1, gid2, nspecial, special, maxspecial, w12,
                              w13, w14);

      // Coul cut
      CoulCutForce_fix(cut_off, alpha, qqr2e, charge_i, charge_j,
    x12, y12, z12, force_coul_factor,energy_coul_factor,
      force_coul, energy_coul);

      force_coul = force_coul-(1-weight)*force_coul_factor;
      energy_coul = energy_coul-(1-weight)*energy_coul_factor;

      // sum force of special_lj_cut  and coul_cut
      force_pair = weight * force_lj + force_coul;
      sum_fx += force_pair * x12;
      sum_fy += force_pair * y12;
      sum_fz += force_pair * z12;
      // sum energy  of special_lj_cut  and coul_cut
      sum_elj += weight * energy_lj;
      sum_ecoul += energy_coul;

      //rbmd::Real local_virial[6];
      rbmd::Real local_virial_xx,local_virial_yy,local_virial_zz,
        local_virial_xy,local_virial_xz,local_virial_yz;
      ComputeVirial(x12, y12, z12,force_pair,
        local_virial_xx,local_virial_yy,local_virial_zz,
        local_virial_xy,local_virial_xz,local_virial_yz);

      //
      sum_virial[0] +=local_virial_xx;
      sum_virial[1] +=local_virial_yy;
      sum_virial[2] +=local_virial_zz;
      sum_virial[3] +=local_virial_xy;
      sum_virial[4] +=local_virial_xz;
      sum_virial[5] +=local_virial_yz;
    }

    fx[tid1] = sum_fx;
    fy[tid1] = sum_fy;
    fz[tid1] = sum_fz;
    //
    for(int i =0;i<6;++i) {
      flat_virial[  i * num_atoms + tid1] = sum_virial[i];
    }
  }

  rbmd::Real block_sum_elj =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_elj)
          .Sum(sum_elj);
  rbmd::Real block_sum_ecoul =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_ecoul)
          .Sum(sum_ecoul);

  if (threadIdx.x == 0) {
    atomicAdd(total_evdwl, block_sum_elj);
    atomicAdd(total_ecoul, block_sum_ecoul);
  }
}

inline __device__ rbmd::Real LJForceFromTypePair(
    rbmd::Real distance_2, rbmd::Real lower_cutoff_2,
    rbmd::Real upper_cutoff_2, rbmd::Real multiplier,
    rbmd::Real lj_force12, rbmd::Real lj_force6) {
  if (distance_2 <= lower_cutoff_2 || distance_2 >= upper_cutoff_2) {
    return rbmd::Real(0);
  }
  const rbmd::Real r2inv = rbmd::Real(1) / distance_2;
  const rbmd::Real r6inv = r2inv * r2inv * r2inv;
  return -multiplier * r2inv * r6inv *
         (lj_force12 * r6inv - lj_force6);
}

//RBL : SpecialLJCutCoul
__global__ void ComputeSpecialLJCutCoulRBLForce(
     Box box, const rbmd::Real rs, const rbmd::Real rc,
    const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
    const rbmd::Id pice_num, const rbmd::Real alpha, const rbmd::Real qqr2e,
    const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Id lj_num_types, const rbmd::Real* lj_force12,
    const rbmd::Real* lj_force6, const rbmd::Id* start_id,
    const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const rbmd::Id* id_random_neighbor, const rbmd::Id* random_neighbor_num,
    const int* nspecial, const rbmd::Id* special, int maxspecial,
    rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
    const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz) {
  rbmd::Real sum_fx = 0;
  rbmd::Real sum_fy = 0;
  rbmd::Real sum_fz = 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    const rbmd::Id gid1 = atoms_id[tid1];
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real charge_i = charge[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];
    // rs
    for (rbmd::Id j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      const rbmd::Id gid2 = atoms_id[tid2];
      rbmd::Id typej = atoms_type[tid2];
      rbmd::Real charge_j = charge[tid2];
      const rbmd::Id pair_index = typei * lj_num_types + typej;
      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real px12 = x2 - x1;
      rbmd::Real py12 = y2 - y1;
      rbmd::Real pz12 = z2 - z1;
      MinImageDistance_while(box, px12, py12, pz12);

      // compute the force_rs
      rbmd::Real force_coul_rs,force_coul_factor_rs;
      rbmd::Real fs_ij;
      const rbmd::Real distance_2 =
          px12 * px12 + py12 * py12 + pz12 * pz12;
      rbmd::Real force_lj_rs = LJForceFromTypePair(
          distance_2, EPSILON, rs * rs, rbmd::Real(1),
          lj_force12[pair_index], lj_force6[pair_index]);

      const rbmd::Real weight =
          LookupSpecialWeight(tid1, gid2, nspecial, special, maxspecial, w12,
                              w13, w14);
      force_lj_rs = weight * force_lj_rs;

      CoulCutForce_rs_fix(rs,alpha,qqr2e,charge_i,charge_j,
        px12, py12, pz12, force_coul_factor_rs,force_coul_rs);
      force_coul_rs = force_coul_rs - (1-weight)*force_coul_factor_rs;

      fs_ij = force_lj_rs + force_coul_rs;
      sum_fx += fs_ij * px12;
      sum_fy += fs_ij * py12;
      sum_fz += fs_ij * pz12;
    }

    // rcs
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
      const rbmd::Id gid2 = atoms_id[tid2];
      rbmd::Id typej = atoms_type[tid2];
      rbmd::Real charge_j = charge[tid2];
      const rbmd::Id pair_index = typei * lj_num_types + typej;
      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real px12 = x2 - x1;
      rbmd::Real py12 = y2 - y1;
      rbmd::Real pz12 = z2 - z1;
      MinImageDistance_while(box, px12, py12, pz12);

      // compute the force_rcs
      rbmd::Real force_coul_rcs,force_coul_factor_rcs;
      rbmd::Real fcs_ij;
      const rbmd::Real distance_2 =
          px12 * px12 + py12 * py12 + pz12 * pz12;
      rbmd::Real force_lj_rcs = LJForceFromTypePair(
          distance_2, rs * rs, rc * rc, static_cast<rbmd::Real>(pice_num),
          lj_force12[pair_index], lj_force6[pair_index]);

      const rbmd::Real weight =
          LookupSpecialWeight(tid1, gid2, nspecial, special, maxspecial, w12,
                              w13, w14);
      force_lj_rcs = weight * force_lj_rcs;

      CoulCutForce_rcs_fix(rc,rs, pice_num, alpha, qqr2e,
      charge_i, charge_j, px12, py12, pz12, force_coul_factor_rcs,
    force_coul_rcs);
      force_coul_rcs = force_coul_rcs - (1-weight)*force_coul_factor_rcs;

      fcs_ij = force_lj_rcs + force_coul_rcs;
      sum_fx += fcs_ij * px12;
      sum_fy += fcs_ij * py12;
      sum_fz += fcs_ij * pz12;
    }

    fx[tid1] = sum_fx;
    fy[tid1] = sum_fy;
    fz[tid1] = sum_fz;
  }
}

//verlet-list : SpecialLJCutCoul Energy
__global__ void ComputeSpecialLJCutCoulEnergy(
     Box box, const rbmd::Real cut_off,
    const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
    const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Real* sigma, const rbmd::Real* eps, const rbmd::Id* start_id,
    const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const int* nspecial, const rbmd::Id* special, int maxspecial,
    rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
    const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz,  rbmd::Real* flat_virial,
    rbmd::Real* total_evdwl, rbmd::Real* total_ecoul) {
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
      temp_storage_elj;
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
      temp_storage_ecoul;
  rbmd::Real sum_elj = 0;
  rbmd::Real sum_ecoul = 0;
  //virial init
  rbmd::Real sum_virial[6];
  for (int i = 0; i < 6; ++i)
  {
    sum_virial[i] = 0.0;
  }

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_atoms) {
    const rbmd::Id gid1 = atoms_id[tid1];
    rbmd::Id typei = atoms_type[tid1];
    rbmd::Real eps_i = eps[typei];
    rbmd::Real sigma_i = sigma[typei];
    rbmd::Real charge_i = charge[tid1];
    rbmd::Real x1 = px[tid1];
    rbmd::Real y1 = py[tid1];
    rbmd::Real z1 = pz[tid1];
    for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
      rbmd::Id tid2 = id_verletlist[j];
      if (tid2 == static_cast<rbmd::Id>(tid1)) {
        continue;
      }
      const rbmd::Id gid2 = atoms_id[tid2];

      rbmd::Id typej = atoms_type[tid2];
      rbmd::Real eps_j = eps[typej];
      rbmd::Real sigma_j = sigma[typej];
      rbmd::Real charge_j = charge[tid2];

      // mix
      rbmd::Real eps_ij = SQRT(eps_i * eps_j);
      rbmd::Real sigma_ij = (sigma_i + sigma_j) / 2;

      rbmd::Real x2 = px[tid2];
      rbmd::Real y2 = py[tid2];
      rbmd::Real z2 = pz[tid2];
      rbmd::Real px12 = x2 - x1;
      rbmd::Real py12 = y2 - y1;
      rbmd::Real pz12 = z2 - z1;
      MinImageDistance_while(box, px12, py12, pz12);

      rbmd::Real force_lj, force_coul, force_pair,force_coul_factor;
      rbmd::Real energy_lj, energy_coul,energy_coul_factor;

      // lj cut
      lj126(cut_off, px12, py12, pz12, eps_ij, sigma_ij, force_lj, energy_lj);

      const rbmd::Real weight =
          LookupSpecialWeight(tid1, gid2, nspecial, special, maxspecial, w12,
                              w13, w14);
      force_lj = weight * force_lj;
      energy_lj = weight * energy_lj;

      // Coul cut
      CoulCutForce_fix(cut_off, alpha, qqr2e, charge_i, charge_j,
        px12, py12, pz12, force_coul_factor,energy_coul_factor,
      force_coul,energy_coul);
      force_coul = force_coul-(1-weight)*force_coul_factor;
      energy_coul = energy_coul- (1-weight)*energy_coul_factor;

      //sum force of special_lj_cut  and coul
      force_pair = force_lj + force_coul;

      //sum energy  of special_lj_cut  and coul
      sum_elj += energy_lj;
      sum_ecoul += energy_coul;


      //rbmd::Real local_virial[6];
      rbmd::Real local_virial_xx,local_virial_yy,local_virial_zz,
  local_virial_xy,local_virial_xz,local_virial_yz;
      ComputeVirial(px12, py12, pz12,force_pair,local_virial_xx,local_virial_yy,
    local_virial_zz,local_virial_xy,local_virial_xz,local_virial_yz);

      //
      sum_virial[0] +=local_virial_xx;
      sum_virial[1] +=local_virial_yy;
      sum_virial[2] +=local_virial_zz;
      sum_virial[3] +=local_virial_xy;
      sum_virial[4] +=local_virial_xz;
      sum_virial[5] +=local_virial_yz;
    }

    //
    for(int i =0;i<6;++i) {
      flat_virial[  i* num_atoms + tid1 ] = sum_virial[i];
    }
  }
  rbmd::Real block_sum_elj =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_elj)
          .Sum(sum_elj);
  rbmd::Real block_sum_ecoul =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_ecoul)
          .Sum(sum_ecoul);

  if (threadIdx.x == 0) {
    atomicAdd(total_evdwl, block_sum_elj);
    atomicAdd(total_ecoul, block_sum_ecoul);
  }
}

 //bond
__global__ void ComputeBondForce(
     Box box, const rbmd::Id num_atoms,const rbmd::Id num_bonds,
     const rbmd::Id* atom_id_to_idx,
    const rbmd::Real* bond_coeffs_k, const rbmd::Real* bond_coeffs_equilibrium,
    const rbmd::Id* bond_type, const rbmd::Id* bondlisti,
    const rbmd::Id* bondlistj, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
    rbmd::Real* flat_virial,rbmd::Real* global_virial,rbmd::Real* energy_bond) {
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
      temp_storage;
  rbmd::Real local_energy_bond = 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_bonds) {
    rbmd::Id bondi = bondlisti[tid1];
    rbmd::Id bondj = bondlistj[tid1];
    rbmd::Id bondii = atom_id_to_idx[bondi];
    rbmd::Id bondjj = atom_id_to_idx[bondj];

    rbmd::Id bondtype = bond_type[tid1];
    rbmd::Real k = bond_coeffs_k[bondtype];
    rbmd::Real equilibrium_bond = bond_coeffs_equilibrium[bondtype];

    rbmd::Real x12 = px[bondii] - px[bondjj];
    rbmd::Real y12 = py[bondii] - py[bondjj];
    rbmd::Real z12 = pz[bondii] - pz[bondjj];
    MinImageDistance_while(box, x12, y12, z12);
    rbmd::Real dis_12 = SQRT(x12 * x12 + y12 * y12 + z12 * z12);
    rbmd::Real dr = dis_12 - equilibrium_bond;
    rbmd::Real rk = k * dr;

    // energy
    local_energy_bond = rk * dr; //double counting??

    rbmd::Real forcebondij;
    if (dis_12 > 0.0) //??
      forcebondij = -2.0 * rk / dis_12;
    else
      forcebondij = 0.0;

    rbmd::Real fx_ij = forcebondij * x12;
    rbmd::Real fy_ij = forcebondij * y12;
    rbmd::Real fz_ij = forcebondij * z12;

    // apply force to each of 2 atoms
    atomicAdd(&fx[bondii], fx_ij);
    atomicAdd(&fy[bondii], fy_ij);
    atomicAdd(&fz[bondii], fz_ij);

    atomicAdd(&fx[bondjj], -fx_ij);
    atomicAdd(&fy[bondjj], -fy_ij);
    atomicAdd(&fz[bondjj], -fz_ij);

    rbmd::Real global_virial_temp[6];
    global_virial_temp[0]  = x12 *x12 *forcebondij;
    global_virial_temp[1]  = y12 *y12 *forcebondij;
    global_virial_temp[2]  = z12 *z12 *forcebondij;
    global_virial_temp[3]  = x12 *y12 *forcebondij;
    global_virial_temp[4]  = x12 *z12 *forcebondij;
    global_virial_temp[5]  = y12 *z12 *forcebondij;

    global_virial[0 * num_bonds +  tid1]= global_virial_temp[0];
    global_virial[1 * num_bonds +  tid1]= global_virial_temp[1];
    global_virial[2 * num_bonds +  tid1]= global_virial_temp[2];
    global_virial[3 * num_bonds +  tid1]= global_virial_temp[3];
    global_virial[4 * num_bonds +  tid1]= global_virial_temp[4];
    global_virial[5 * num_bonds +  tid1]= global_virial_temp[5];

    rbmd::Real local_virial[6];
    local_virial[0]  = 0.5 *global_virial_temp[0]; //double counting
    local_virial[1]  = 0.5 *global_virial_temp[1];
    local_virial[2]  = 0.5 *global_virial_temp[2];
    local_virial[3]  = 0.5 *global_virial_temp[3];
    local_virial[4]  = 0.5 *global_virial_temp[4];
    local_virial[5]  = 0.5 *global_virial_temp[5];

    //
    atomicAdd(&flat_virial[0 * num_atoms + bondii], local_virial[0]);
    atomicAdd(&flat_virial[1 * num_atoms + bondii], local_virial[1]);
    atomicAdd(&flat_virial[2 * num_atoms + bondii], local_virial[2]);
    atomicAdd(&flat_virial[3 * num_atoms + bondii], local_virial[3]);
    atomicAdd(&flat_virial[4 * num_atoms + bondii], local_virial[4]);
    atomicAdd(&flat_virial[5 * num_atoms + bondii], local_virial[5]);

    atomicAdd(&flat_virial[0 * num_atoms + bondjj], local_virial[0]);
    atomicAdd(&flat_virial[1 * num_atoms + bondjj], local_virial[1]);
    atomicAdd(&flat_virial[2 * num_atoms + bondjj], local_virial[2]);
    atomicAdd(&flat_virial[3 * num_atoms + bondjj], local_virial[3]);
    atomicAdd(&flat_virial[4 * num_atoms + bondjj], local_virial[4]);
    atomicAdd(&flat_virial[5 * num_atoms + bondjj], local_virial[5]);
  }
  rbmd::Real block_sum =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
          .Sum(local_energy_bond);

  if (threadIdx.x == 0) {
    atomicAdd(energy_bond, block_sum);
  }
}

//angle
  __global__ void ComputeAngleForce(
       Box box, const rbmd::Id num_atoms,const rbmd::Id num_anglels,
       const rbmd::Id* atom_id_to_idx,
      const rbmd::Real* anglel_coeffs_k,
      const rbmd::Real* anglel_coeffs_equilibrium, const rbmd::Id* anglel_type,
      const rbmd::Id* anglelisti, const rbmd::Id* anglelistj,
      const rbmd::Id* anglelistk, const rbmd::Real* px, const rbmd::Real* py,
      const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* flat_virial,rbmd::Real* global_virial,rbmd::Real* energy_angle) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage;
    rbmd::Real local_energy_angle = 0;

    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_anglels) {
      rbmd::Id angleli = anglelisti[tid1];
      rbmd::Id anglelj = anglelistj[tid1];
      rbmd::Id anglelk = anglelistk[tid1];
      rbmd::Id anglelii = atom_id_to_idx[angleli];
      rbmd::Id angleljj = atom_id_to_idx[anglelj];
      rbmd::Id anglelkk = atom_id_to_idx[anglelk];

      rbmd::Id angleltype = anglel_type[tid1];
      rbmd::Real k = anglel_coeffs_k[angleltype];
      rbmd::Real equilibrium_angle = anglel_coeffs_equilibrium[angleltype];

      rbmd::Real x12 = px[anglelii] - px[angleljj];  // i j
      rbmd::Real y12 = py[anglelii] - py[angleljj];
      rbmd::Real z12 = pz[anglelii] - pz[angleljj];
      MinImageDistance_while(box, x12, y12, z12);

      rbmd::Real x23 = px[anglelkk] - px[angleljj];  // k j
      rbmd::Real y23 = py[anglelkk] - py[angleljj];
      rbmd::Real z23 = pz[anglelkk] - pz[angleljj];
      MinImageDistance_while(box, x23, y23, z23);

      rbmd::Real dis_12_2 = x12 * x12 + y12 * y12 + z12 * z12;
      rbmd::Real dis_12 = SQRT(dis_12_2);

      rbmd::Real dis_23_2 = x23 * x23 + y23 * y23 + z23 * z23;
      rbmd::Real dis_23 = SQRT(dis_23_2);

      rbmd::Real cosangle = x12 * x23 + y12 * y23 + z12 * z23;
      cosangle /= dis_12 * dis_23;

      if (cosangle > 1.0) cosangle = 1.0;
      if (cosangle < -1.0) cosangle = -1.0;
      rbmd::Real s = SQRT(1.0 - cosangle * cosangle);

      if (s < SMALL) s = SMALL;
      s = 1.0 / s;

      rbmd::Real dtheta = ACOS(cosangle) - (equilibrium_angle * M_PI) / 180;
      rbmd::Real tk = k * dtheta;

      // energy
      local_energy_angle = tk * dtheta;

      rbmd::Real a = -2.0 * tk * s;
      rbmd::Real a11 = a * cosangle / dis_12_2;
      rbmd::Real a12 = -a / (dis_12 * dis_23);
      rbmd::Real a22 = a * cosangle / dis_23_2;

      rbmd::Real force_anglei_x, force_anglei_y, force_anglei_z;
      rbmd::Real force_anglek_x, force_anglek_y, force_anglek_z;
      rbmd::Real force_anglej_x, force_anglej_y, force_anglej_z;

      force_anglei_x = a11 * x12 + a12 * x23;
      force_anglei_y = a11 * y12 + a12 * y23;
      force_anglei_z = a11 * z12 + a12 * z23;

      force_anglek_x = a22 * x23 + a12 * x12;
      force_anglek_y = a22 * y23 + a12 * y12;
      force_anglek_z = a22 * z23 + a12 * z12;

      force_anglej_x = -(force_anglei_x + force_anglek_x);
      force_anglej_y = -(force_anglei_y + force_anglek_y);
      force_anglej_z = -(force_anglei_z + force_anglek_z);

      atomicAdd(&fx[anglelii], force_anglei_x);
      atomicAdd(&fy[anglelii], force_anglei_y);
      atomicAdd(&fz[anglelii], force_anglei_z);

      atomicAdd(&fx[anglelkk], force_anglek_x);
      atomicAdd(&fy[anglelkk], force_anglek_y);
      atomicAdd(&fz[anglelkk], force_anglek_z);

      atomicAdd(&fx[angleljj], force_anglej_x);
      atomicAdd(&fy[angleljj], force_anglej_y);
      atomicAdd(&fz[angleljj], force_anglej_z);

      //virial
      rbmd::Real global_virial_temp[6];
      global_virial_temp[0]  = (x12 * force_anglei_x + x23 * force_anglek_x);
      global_virial_temp[1]  = (y12 * force_anglei_y + y23 * force_anglek_y);
      global_virial_temp[2]  = (z12 * force_anglei_z + z23 * force_anglek_z);
      global_virial_temp[3]  = (x12 * force_anglei_y + x23 * force_anglek_y);
      global_virial_temp[4]  = (x12 * force_anglei_z + x23 * force_anglek_z);
      global_virial_temp[5]  = (y12 * force_anglei_z + y23 * force_anglek_z);
      global_virial[0+num_anglels + tid1] = global_virial_temp[0];
      global_virial[1+num_anglels + tid1] = global_virial_temp[1];
      global_virial[2+num_anglels + tid1] = global_virial_temp[2];
      global_virial[3+num_anglels + tid1] = global_virial_temp[3];
      global_virial[4+num_anglels + tid1] = global_virial_temp[4];
      global_virial[5+num_anglels + tid1] = global_virial_temp[5];

      rbmd::Real local_virial[6];
      local_virial[0]  =  0.3333333333*global_virial_temp[0];
      local_virial[1]  =  0.3333333333*global_virial_temp[1];
      local_virial[2]  =  0.3333333333*global_virial_temp[2];
      local_virial[3]  =  0.3333333333*global_virial_temp[3];
      local_virial[4]  =  0.3333333333*global_virial_temp[4];
      local_virial[5]  =  0.3333333333*global_virial_temp[5];

      //
      //Column-Major Order : j * M + i
      atomicAdd(&flat_virial[0 * num_atoms + anglelii], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + anglelii], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + anglelii], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + anglelii], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + anglelii], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + anglelii], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + angleljj], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + angleljj], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + angleljj], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + angleljj], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + angleljj], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + angleljj], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + anglelkk], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + anglelkk], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + anglelkk], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + anglelkk], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + anglelkk], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + anglelkk], local_virial[5]);

    }

    rbmd::Real block_sum =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
            .Sum(local_energy_angle);

    if (threadIdx.x == 0) {
      atomicAdd(energy_angle, block_sum);
    }
  }

  //Dihedral
  __global__ void ComputeDihedralForce(
       Box box, const rbmd::Id num_atoms,const rbmd::Id num_dihedrals,
       const rbmd::Id* atom_id_to_idx,const rbmd::Real* dihedral_coeffs_k,
       const rbmd::Id* dihedral_coeffs_sign,
      const rbmd::Id* dihedral_coeffs_multiplicity, const rbmd::Id* dihedral_type,
      const rbmd::Id* dihedrallisti, const rbmd::Id* dihedrallistj,
      const rbmd::Id* dihedrallistk, const rbmd::Id* dihedrallistw,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,rbmd::Real* flat_virial,
      rbmd::Real* global_virial,rbmd::Real* energy_dihedral) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage;
    rbmd::Real local_energy_dihedral = 0;

    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_dihedrals) {
      rbmd::Id dihedrali = dihedrallisti[tid1];
      rbmd::Id dihedralj = dihedrallistj[tid1];
      rbmd::Id dihedralk = dihedrallistk[tid1];
      rbmd::Id dihedralw = dihedrallistw[tid1];

      rbmd::Id dihedralii = atom_id_to_idx[dihedrali];
      rbmd::Id dihedraljj = atom_id_to_idx[dihedralj];
      rbmd::Id dihedralkk = atom_id_to_idx[dihedralk];
      rbmd::Id dihedralww = atom_id_to_idx[dihedralw];
      rbmd::Real cos_shift, sin_shift;

      rbmd::Id dihedraltype = dihedral_type[tid1];
      if (dihedral_coeffs_sign[dihedraltype] == 1) {
        cos_shift = 1.0;
        sin_shift = 0.0;
      } else {
        cos_shift = -1.0;
        sin_shift = 0.0;
      }

      rbmd::Real k = dihedral_coeffs_k[dihedraltype];
      rbmd::Real x12 = px[dihedralii] - px[dihedraljj];  // i j =vb1
      rbmd::Real y12 = py[dihedralii] - py[dihedraljj];
      rbmd::Real z12 = pz[dihedralii] - pz[dihedraljj];
      MinImageDistance_while(box, x12, y12, z12);

      rbmd::Real x23 = px[dihedralkk] - px[dihedraljj];  //  k j=vb2
      rbmd::Real y23 = py[dihedralkk] - py[dihedraljj];
      rbmd::Real z23 = pz[dihedralkk] - pz[dihedraljj];
      MinImageDistance_while(box, x23, y23, z23);

      rbmd::Real x23m = -x23;  // =vb2m
      rbmd::Real y23m = -y23;
      rbmd::Real z23m = -z23;

      rbmd::Real x34 = px[dihedralww] - px[dihedralkk];  // w k   =vb3
      rbmd::Real y34 = py[dihedralww] - py[dihedralkk];
      rbmd::Real z34 = pz[dihedralww] - pz[dihedralkk];
      MinImageDistance_while(box, x34, y34, z34);
      // c,s calculation

      rbmd::Real ax = y12 * z23m - z12 * y23m;
      rbmd::Real ay = z12 * x23m - x12 * z23m;
      rbmd::Real az = x12 * y23m - y12 * x23m;
      rbmd::Real bx = y34 * z23m - z34 * y23m;
      rbmd::Real by = z34 * x23m - x34 * z23m;
      rbmd::Real bz = x34 * y23m - y34 * x23m; //fix :y34
      rbmd::Real rasq = ax * ax + ay * ay + az * az;
      rbmd::Real rbsq = bx * bx + by * by + bz * bz;
      rbmd::Real rgsq = x23m * x23m + y23m * y23m + z23m * z23m;
      rbmd::Real rg = SQRT(rgsq);

      rbmd::Real rginv, ra2inv, rb2inv;
      rginv = ra2inv = rb2inv = 0.0;
      if (rg > 0) rginv = 1.0 / rg;

      if (rasq > 0) ra2inv = 1.0 / rasq;

      if (rbsq > 0) rb2inv = 1.0 / rbsq;

      rbmd::Real rabinv = SQRT(ra2inv * rb2inv);

      rbmd::Real c = (ax * bx + ay * by + az * bz) * rabinv;
      rbmd::Real s = rg * rabinv * (ax * x34 + ay * y34 + az * z34);

      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;

      rbmd::Id m = dihedral_coeffs_multiplicity[dihedraltype];
      rbmd::Real p = 1.0;
      rbmd::Real ddf1, df1;
      ddf1 = df1 = 0.0;
      for (rbmd::Id i = 0; i < m; i++) {
        ddf1 = p * c - df1 * s;
        df1 = p * s + df1 * c;
        p = ddf1;
      }

      p = p * cos_shift + df1 * sin_shift;
      df1 = df1 * cos_shift - ddf1 * sin_shift;
      df1 *= -m;
      p += 1.0;

      if (m == 0) {
        p = 1.0 + cos_shift;
        // p = 1.0 + cos_shift[type];
        df1 = 0.0;
      }

      // energy
      local_energy_dihedral = k * p;

      rbmd::Real fg = x12 * x23m + y12 * y23m + z12 * z23m;
      rbmd::Real hg = x34 * x23m + y34 * y23m + z34 * z23m;

      rbmd::Real fga = fg * ra2inv * rginv;
      rbmd::Real hgb = hg * rb2inv * rginv;
      rbmd::Real gaa = -ra2inv * rg;
      rbmd::Real gbb = rb2inv * rg;

      rbmd::Real dtfx = gaa * ax;
      rbmd::Real dtfy = gaa * ay;
      rbmd::Real dtfz = gaa * az;
      rbmd::Real dtgx = fga * ax - hgb * bx;
      rbmd::Real dtgy = fga * ay - hgb * by;
      rbmd::Real dtgz = fga * az - hgb * bz;
      rbmd::Real dthx = gbb * bx;
      rbmd::Real dthy = gbb * by;
      rbmd::Real dthz = gbb * bz;

      rbmd::Real df = -k * df1;
      // df = -k[type] * df1;
      rbmd::Real sx2 = df * dtgx;
      rbmd::Real sy2 = df * dtgy;
      rbmd::Real sz2 = df * dtgz;

      // force
      rbmd::Real force_dihedrali_x, force_dihedrali_y, force_dihedrali_z;
      rbmd::Real force_dihedralj_x, force_dihedralj_y, force_dihedralj_z;
      rbmd::Real force_dihedralk_x, force_dihedralk_y, force_dihedralk_z;
      rbmd::Real force_dihedralw_x, force_dihedralw_y, force_dihedralw_z;

      force_dihedrali_x = df * dtfx;
      force_dihedrali_y = df * dtfy;
      force_dihedrali_z = df * dtfz;

      force_dihedralj_x = sx2 - force_dihedrali_x;
      force_dihedralj_y = sy2 - force_dihedrali_y;
      force_dihedralj_z = sz2 - force_dihedrali_z;

      force_dihedralw_x = df * dthx;
      force_dihedralw_y = df * dthy;
      force_dihedralw_z = df * dthz;

      force_dihedralk_x = -sx2 - force_dihedralw_x;
      force_dihedralk_y = -sy2 - force_dihedralw_y;
      force_dihedralk_z = -sz2 - force_dihedralw_z;

      atomicAdd(&fx[dihedralii], force_dihedrali_x);
      atomicAdd(&fy[dihedralii], force_dihedrali_y);
      atomicAdd(&fz[dihedralii], force_dihedrali_z);

      atomicAdd(&fx[dihedraljj], force_dihedralj_x);
      atomicAdd(&fy[dihedraljj], force_dihedralj_y);
      atomicAdd(&fz[dihedraljj], force_dihedralj_z);

      atomicAdd(&fx[dihedralww], force_dihedralw_x);
      atomicAdd(&fy[dihedralww], force_dihedralw_y);
      atomicAdd(&fz[dihedralww], force_dihedralw_z);

      atomicAdd(&fx[dihedralkk], force_dihedralk_x);
      atomicAdd(&fy[dihedralkk], force_dihedralk_y);
      atomicAdd(&fz[dihedralkk], force_dihedralk_z);

      rbmd::Real  global_virial_temp[6];
      global_virial_temp[0] = (x12 * force_dihedrali_x + x23 * force_dihedralk_x +
      (x34 + x23) * force_dihedralw_x);

      global_virial_temp[1] =  (y12 * force_dihedrali_y + y23 * force_dihedralk_y +
         (y34 + y23) * force_dihedralw_y);

      global_virial_temp[2] = (z12 * force_dihedrali_z + z23 * force_dihedralk_z +
         (z34 + z23) * force_dihedralw_z);

      global_virial_temp[3] = (x12 * force_dihedrali_y + x23 * force_dihedralk_y +
         (x34 + x23) * force_dihedralw_y);

      global_virial_temp[4] = (x12* force_dihedrali_z + x23 * force_dihedralk_z +
         (x34 + x23) * force_dihedralw_z);

      global_virial_temp[5] = (y12 * force_dihedrali_z + y23 * force_dihedralk_z +
         (y34 + y23) * force_dihedralw_z);

      global_virial[0 * num_dihedrals + tid1] = global_virial_temp[0];
      global_virial[1 * num_dihedrals + tid1] = global_virial_temp[1];
      global_virial[2 * num_dihedrals + tid1] = global_virial_temp[2];
      global_virial[3 * num_dihedrals + tid1] = global_virial_temp[3];
      global_virial[4 * num_dihedrals + tid1] = global_virial_temp[4];
      global_virial[5 * num_dihedrals + tid1] = global_virial_temp[5];

      rbmd::Real local_virial[6];
      local_virial[0] = 0.25* global_virial_temp[0];
      local_virial[1] = 0.25* global_virial_temp[1];
      local_virial[2] = 0.25* global_virial_temp[2];
      local_virial[3] = 0.25* global_virial_temp[3];
      local_virial[4] = 0.25* global_virial_temp[4];
      local_virial[5] = 0.25* global_virial_temp[5];

      //
      atomicAdd(&flat_virial[0 * num_atoms + dihedralii], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedralii], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedralii], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedralii], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedralii], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedralii], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + dihedraljj], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedraljj], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedraljj], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedraljj], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedraljj], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedraljj], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + dihedralkk], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedralkk], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedralkk], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedralkk], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedralkk], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedralkk], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + dihedralww], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedralww], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedralww], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedralww], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedralww], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedralww], local_virial[5]);
    }
    rbmd::Real block_sum =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
            .Sum(local_energy_dihedral);

    if (threadIdx.x == 0) {
      atomicAdd(energy_dihedral, block_sum);
    }
  }

  __global__ void ComputeDihedralOPLSForce(
       Box box, const rbmd::Id num_atoms,const rbmd::Id num_dihedrals,
       const rbmd::Id* atom_id_to_idx,const rbmd::Real* dihedral_coeffs_k1,
       const rbmd::Real* dihedral_coeffs_k2,const rbmd::Real* dihedral_coeffs_k3,
      const rbmd::Real* dihedral_coeffs_k4, const rbmd::Id* dihedral_type,
      const rbmd::Id* dihedrallisti, const rbmd::Id* dihedrallistj,
      const rbmd::Id* dihedrallistk, const rbmd::Id* dihedrallistw,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,rbmd::Real* flat_virial,
      rbmd::Real* global_virial,rbmd::Real* energy_dihedral) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage;
    rbmd::Real local_energy_dihedral = 0;

    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_dihedrals) {
      rbmd::Id dihedrali = dihedrallisti[tid1];
      rbmd::Id dihedralj = dihedrallistj[tid1];
      rbmd::Id dihedralk = dihedrallistk[tid1];
      rbmd::Id dihedralw = dihedrallistw[tid1];

      rbmd::Id dihedralii = atom_id_to_idx[dihedrali];
      rbmd::Id dihedraljj = atom_id_to_idx[dihedralj];
      rbmd::Id dihedralkk = atom_id_to_idx[dihedralk];
      rbmd::Id dihedralww = atom_id_to_idx[dihedralw];

      rbmd::Id dihedraltype = dihedral_type[tid1];

      rbmd::Real k1 = dihedral_coeffs_k1[dihedraltype];
      rbmd::Real k2 = dihedral_coeffs_k2[dihedraltype];
      rbmd::Real k3 = dihedral_coeffs_k3[dihedraltype];
      rbmd::Real k4 = dihedral_coeffs_k4[dihedraltype];
      rbmd::Real x12 = px[dihedralii] - px[dihedraljj];  // i j =vb1
      rbmd::Real y12 = py[dihedralii] - py[dihedraljj];
      rbmd::Real z12 = pz[dihedralii] - pz[dihedraljj];
      MinImageDistance_while(box, x12, y12, z12);

      rbmd::Real x23 = px[dihedralkk] - px[dihedraljj];  //  k j=vb2
      rbmd::Real y23 = py[dihedralkk] - py[dihedraljj];
      rbmd::Real z23 = pz[dihedralkk] - pz[dihedraljj];
      MinImageDistance_while(box, x23, y23, z23);

      rbmd::Real x23m = -x23;  // =vb2m
      rbmd::Real y23m = -y23;
      rbmd::Real z23m = -z23;

      rbmd::Real x34 = px[dihedralww] - px[dihedralkk];  // w k   =vb3
      rbmd::Real y34 = py[dihedralww] - py[dihedralkk];
      rbmd::Real z34 = pz[dihedralww] - pz[dihedralkk];
      MinImageDistance_while(box, x34, y34, z34);

      // c0 calculation
      rbmd::Real sb1,sb2,sb3,rb1,rb3,c0,b1mag2, b1mag, b2mag2;
      rbmd::Real b2mag, b3mag2, b3mag, ctmp, r12c1, c1mag, r12c2;
      rbmd::Real c2mag, sc1, sc2, s1, s12, c, p, pd, a, a11, a22;
      rbmd::Real  a33, a12, a13, a23, sx2, sy2, sz2;
      rbmd::Real  s2, cx, cy, cz, cmag, dx, phi, si, siinv, sin2;

      sb1 = 1.0 / (x12 * x12 + y12 * y12 + z12 * z12);
      sb2 = 1.0 / (x23 * x23 + y23 * y23 + z23 * z23);
      sb3 = 1.0 / (x34 * x34 + y34 * y34 + z34 * z34);

      rb1 = SQRT(sb1);
      rb3 = SQRT(sb3);

      c0 = (x12 * x34 + y12 * y34 + z12 * z34) * rb1 * rb3;

      // 1st and 2nd angle

      b1mag2 = x12 * x12 + y12 * y12 + z12 * z12;
      b1mag = SQRT(b1mag2);
      b2mag2 = x23 * x23 + y23 * y23 + z23 * z23;
      b2mag = SQRT(b2mag2);
      b3mag2 = x34 * x34 + y34 * y34 + z34 * z34;
      b3mag = SQRT(b3mag2);

      ctmp = x12 * x23 + y12 * y23 + z12 * z23;
      r12c1 = 1.0 / (b1mag * b2mag);
      c1mag = ctmp * r12c1;

      ctmp = x23m * x34 + y23m * y34 + z23m * z34;
      r12c2 = 1.0 / (b2mag * b3mag);
      c2mag = ctmp * r12c2;

      // cos and sin of 2 angles and final c

      sin2 = MAX(1.0 - c1mag * c1mag, 0.0);
      sc1 = SQRT(sin2);
      if (sc1 < SMALL) sc1 = SMALL;
      sc1 = 1.0 / sc1;

      sin2 = MAX(1.0 - c2mag * c2mag, 0.0);
      sc2 = SQRT(sin2);
      if (sc2 < SMALL) sc2 = SMALL;
      sc2 = 1.0 / sc2;

      s1 = sc1 * sc1;
      s2 = sc2 * sc2;
      s12 = sc1 * sc2;
      c = (c0 + c1mag * c2mag) * s12;

      cx = y12 * z23 - z12 * y23;
      cy = z12 * x23 - x12 * z23;
      cz = x12 * y23 - y12 * x23;
      cmag = SQRT(cx * cx + cy * cy + cz * cz);
      dx = (cx * x34 + cy * y34 + cz * z34) / cmag / b3mag;
      // error check

      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;


      phi = ACOS(c);
      if (dx < 0.0) phi *= -1.0;
      si = SIN(phi);
      if (fabs(si) < SMALLER) si = SMALLER;
      siinv = 1.0 / si;

      p = k1 * (1.0 + c) + k2 * (1.0 - COS(2.0 * phi)) +
          k3 * (1.0 + COS(3.0 * phi)) + k4 * (1.0 - COS(4.0 * phi));
      pd = k1 - 2.0 * k2* SIN(2.0 * phi) * siinv +
          3.0 * k3 * SIN(3.0 * phi) * siinv - 4.0 * k4 * SIN(4.0 * phi) * siinv;

      local_energy_dihedral = p;

      a = pd;
      c = c * a;
      s12 = s12 * a;
      a11 = c * sb1 * s1;
      a22 = -sb2 * (2.0 * c0 * s12 - c * (s1 + s2));
      a33 = c * sb3 * s2;
      a12 = -r12c1 * (c1mag * c * s1 + c2mag * s12);
      a13 = -rb1 * rb3 * s12;
      a23 = r12c2 * (c2mag * c * s2 + c1mag * s12);

      sx2 = a12 * x12 + a22 * x23 + a23 * x34;
      sy2 = a12 * y12 + a22 * y23 + a23 * y34;
      sz2 = a12 * z12 + a22 * z23 + a23 * z34;

      // force
      rbmd::Real force_dihedrali_x, force_dihedrali_y, force_dihedrali_z;
      rbmd::Real force_dihedralj_x, force_dihedralj_y, force_dihedralj_z;
      rbmd::Real force_dihedralk_x, force_dihedralk_y, force_dihedralk_z;
      rbmd::Real force_dihedralw_x, force_dihedralw_y, force_dihedralw_z;

      force_dihedrali_x = a11 * x12 + a12 * x23 + a13 * x34;
      force_dihedrali_y = a11 * y12 + a12 * y23 + a13 * y34;
      force_dihedrali_z = a11 * z12 + a12 * z23 + a13 * z34;

      force_dihedralj_x = -sx2 - force_dihedrali_x;
      force_dihedralj_y = -sy2 - force_dihedrali_y;
      force_dihedralj_z = -sz2 - force_dihedrali_z;

      force_dihedralw_x = a13 * x12 + a23 * x23 + a33 * x34;
      force_dihedralw_y = a13 * y12 + a23 * y23 + a33 * y34;
      force_dihedralw_z = a13 * z12 + a23 * z23 + a33 * z34;

      force_dihedralk_x = sx2 - force_dihedralw_x;
      force_dihedralk_y = sy2 - force_dihedralw_y;
      force_dihedralk_z = sz2 - force_dihedralw_z;

      atomicAdd(&fx[dihedralii], force_dihedrali_x);
      atomicAdd(&fy[dihedralii], force_dihedrali_y);
      atomicAdd(&fz[dihedralii], force_dihedrali_z);

      atomicAdd(&fx[dihedraljj], force_dihedralj_x);
      atomicAdd(&fy[dihedraljj], force_dihedralj_y);
      atomicAdd(&fz[dihedraljj], force_dihedralj_z);

      atomicAdd(&fx[dihedralww], force_dihedralw_x);
      atomicAdd(&fy[dihedralww], force_dihedralw_y);
      atomicAdd(&fz[dihedralww], force_dihedralw_z);

      atomicAdd(&fx[dihedralkk], force_dihedralk_x);
      atomicAdd(&fy[dihedralkk], force_dihedralk_y);
      atomicAdd(&fz[dihedralkk], force_dihedralk_z);

      rbmd::Real  global_virial_temp[6];
      global_virial_temp[0] = (x12 * force_dihedrali_x + x23 * force_dihedralk_x +
      (x34 + x23) * force_dihedralw_x);

      global_virial_temp[1] =  (y12 * force_dihedrali_y + y23 * force_dihedralk_y +
         (y34 + y23) * force_dihedralw_y);

      global_virial_temp[2] = (z12 * force_dihedrali_z + z23 * force_dihedralk_z +
         (z34 + z23) * force_dihedralw_z);

      global_virial_temp[3] = (x12 * force_dihedrali_y + x23 * force_dihedralk_y +
         (x34 + x23) * force_dihedralw_y);

      global_virial_temp[4] = (x12* force_dihedrali_z + x23 * force_dihedralk_z +
         (x34 + x23) * force_dihedralw_z);

      global_virial_temp[5] = (y12 * force_dihedrali_z + y23 * force_dihedralk_z +
         (y34 + y23) * force_dihedralw_z);

      global_virial[0 * num_dihedrals + tid1] = global_virial_temp[0];
      global_virial[1 * num_dihedrals + tid1] = global_virial_temp[1];
      global_virial[2 * num_dihedrals + tid1] = global_virial_temp[2];
      global_virial[3 * num_dihedrals + tid1] = global_virial_temp[3];
      global_virial[4 * num_dihedrals + tid1] = global_virial_temp[4];
      global_virial[5 * num_dihedrals + tid1] = global_virial_temp[5];

      rbmd::Real local_virial[6];
      local_virial[0] = 0.25* global_virial_temp[0];
      local_virial[1] = 0.25* global_virial_temp[1];
      local_virial[2] = 0.25* global_virial_temp[2];
      local_virial[3] = 0.25* global_virial_temp[3];
      local_virial[4] = 0.25* global_virial_temp[4];
      local_virial[5] = 0.25* global_virial_temp[5];

      //
      atomicAdd(&flat_virial[0 * num_atoms + dihedralii], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedralii], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedralii], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedralii], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedralii], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedralii], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + dihedraljj], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedraljj], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedraljj], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedraljj], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedraljj], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedraljj], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + dihedralkk], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedralkk], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedralkk], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedralkk], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedralkk], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedralkk], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + dihedralww], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + dihedralww], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + dihedralww], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + dihedralww], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + dihedralww], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + dihedralww], local_virial[5]);
    }
    rbmd::Real block_sum =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
            .Sum(local_energy_dihedral);

    if (threadIdx.x == 0) {
      atomicAdd(energy_dihedral, block_sum);
    }
  }

__global__ void ComputeDihedralFourierForce(
    Box box, const rbmd::Id num_atoms, const rbmd::Id num_dihedrals,
    const rbmd::Id* atom_id_to_idx,
    const rbmd::Id* nterms, const rbmd::Id* fourier_offsets,
    const rbmd::Real* fourier_k, const rbmd::Id* fourier_multiplicity,
    const rbmd::Real* fourier_cos_shift, const rbmd::Real* fourier_sin_shift,
    const rbmd::Id* dihedral_type, const rbmd::Id* dihedrallisti,
    const rbmd::Id* dihedrallistj, const rbmd::Id* dihedrallistk,
    const rbmd::Id* dihedrallistw, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz,rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
    rbmd::Real* flat_virial, rbmd::Real* global_virial,
    rbmd::Real* energy_dihedral) {
  // Shared memory for block-level energy reduction
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
      temp_storage;
  rbmd::Real local_energy_dihedral = 0;

  unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid1 < num_dihedrals) {
    // 1. Fetch atom indices and type
    rbmd::Id i1_id = dihedrallisti[tid1];
    rbmd::Id i2_id = dihedrallistj[tid1];
    rbmd::Id i3_id = dihedrallistk[tid1];
    rbmd::Id i4_id = dihedrallistw[tid1];

    rbmd::Id i1 = atom_id_to_idx[i1_id];
    rbmd::Id i2 = atom_id_to_idx[i2_id];
    rbmd::Id i3 = atom_id_to_idx[i3_id];
    rbmd::Id i4 = atom_id_to_idx[i4_id];

    rbmd::Id type = dihedral_type[tid1];

    // 2. Geometric setup
    rbmd::Real vb1x = px[i1] - px[i2];
    rbmd::Real vb1y = py[i1] - py[i2];
    rbmd::Real vb1z = pz[i1] - pz[i2];
    MinImageDistance_while(box, vb1x, vb1y, vb1z);

    rbmd::Real vb2x = px[i3] - px[i2];
    rbmd::Real vb2y = py[i3] - py[i2];
    rbmd::Real vb2z = pz[i3] - pz[i2];
    MinImageDistance_while(box, vb2x, vb2y, vb2z);

    rbmd::Real vb3x = px[i4] - px[i3];
    rbmd::Real vb3y = py[i4] - py[i3];
    rbmd::Real vb3z = pz[i4] - pz[i3];
    MinImageDistance_while(box, vb3x, vb3y, vb3z);

    rbmd::Real vb2xm = -vb2x;
    rbmd::Real vb2ym = -vb2y;
    rbmd::Real vb2zm = -vb2z;

    rbmd::Real ax = vb1y * vb2zm - vb1z * vb2ym;
    rbmd::Real ay = vb1z * vb2xm - vb1x * vb2zm;
    rbmd::Real az = vb1x * vb2ym - vb1y * vb2xm;
    rbmd::Real bx = vb3y * vb2zm - vb3z * vb2ym;
    rbmd::Real by = vb3z * vb2xm - vb3x * vb2zm;
    rbmd::Real bz = vb3x * vb2ym - vb3y * vb2xm;

    rbmd::Real rasq = ax * ax + ay * ay + az * az;
    rbmd::Real rbsq = bx * bx + by * by + bz * bz;
    rbmd::Real rgsq = vb2xm * vb2xm + vb2ym * vb2ym + vb2zm * vb2zm;
    rbmd::Real rg = SQRT(rgsq);

    rbmd::Real rginv = 0.0, ra2inv = 0.0, rb2inv = 0.0;
    if (rg > 0) rginv = 1.0 / rg;
    if (rasq > 0) ra2inv = 1.0 / rasq;
    if (rbsq > 0) rb2inv = 1.0 / rbsq;
    rbmd::Real rabinv = SQRT(ra2inv * rb2inv);

    rbmd::Real c = (ax * bx + ay * by + az * bz) * rabinv; // cos(phi)
    rbmd::Real s = rg * rabinv * (ax * vb3x + ay * vb3y + az * vb3z); // sin(phi)

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    // 3. Loop over Fourier terms to calculate energy and force derivative
    rbmd::Real edihedral = 0.0;
    rbmd::Real df = 0.0; // dE/d(phi)

    rbmd::Id num_terms_for_type = nterms[type];
    rbmd::Id offset = fourier_offsets[type];

    for (rbmd::Id j = 0; j < num_terms_for_type; j++) {
      rbmd::Id term_idx = offset + j;
      rbmd::Real k_j = fourier_k[term_idx];
      rbmd::Id m_j = fourier_multiplicity[term_idx];
      rbmd::Real cos_shift_j = fourier_cos_shift[term_idx];
      rbmd::Real sin_shift_j = fourier_sin_shift[term_idx];

      rbmd::Real p_ = 1.0;    // will become cos(m*phi)
      rbmd::Real df1_ = 0.0;   // will become sin(m*phi)
      rbmd::Real ddf1_ = 0.0;

      // Iteratively find cos(m*phi) and sin(m*phi)
      for (rbmd::Id i = 0; i < m_j; i++) {
        ddf1_ = p_ * c - df1_ * s;
        df1_ = p_ * s + df1_ * c;
        p_ = ddf1_;
      }

      // Apply phase shift: cos(m*phi - d) = cos(m*phi)cos(d) + sin(m*phi)sin(d)
      rbmd::Real p_shifted = p_ * cos_shift_j + df1_ * sin_shift_j;
      // d(cos(m*phi-d))/d(phi) = -m*sin(m*phi-d)
      // sin(m*phi-d) = sin(m*phi)cos(d) - cos(m*phi)sin(d)
      rbmd::Real df1_shifted = df1_ * cos_shift_j - p_ * sin_shift_j;
      df1_shifted *= m_j;

      if (m_j == 0) {
        p_shifted = cos_shift_j;
        df1_shifted = 0.0;
      }

      edihedral += k_j * (1.0 + p_shifted);
      df += k_j * df1_shifted;
    }

    // In the formula, force derivative is -dE/d(phi). We calculated dE/d(phi).
    // The force projection part in LAMMPS uses df = -dE/d(phi).
    // But their code defines df as positive and sx2 = -df * ..., making it correct.
    // Here we must be careful. The LAMMPS code's `df` is d(potential)/d(cos(phi)).
    // Let's re-verify. `df` accumulates `-k * df1_`. And `df1_` is `-m * sin(m*phi-d)`.
    // So `df` accumulates `k * m * sin(m*phi-d)`, which is `dE/d(phi)`.
    // The final force terms are `df * dtf...`. Let's use `df_final = -df`.
    df = -df;

    local_energy_dihedral = edihedral;

    // 4. Project force onto atoms (same logic as LAMMPS)
    rbmd::Real fg = vb1x * vb2xm + vb1y * vb2ym + vb1z * vb2zm;
    rbmd::Real hg = vb3x * vb2xm + vb3y * vb2ym + vb3z * vb2zm;

    rbmd::Real fga = fg * ra2inv * rginv;
    rbmd::Real hgb = hg * rb2inv * rginv;
    rbmd::Real gaa = -ra2inv * rg;
    rbmd::Real gbb = rb2inv * rg;

    rbmd::Real dtfx = gaa * ax;
    rbmd::Real dtfy = gaa * ay;
    rbmd::Real dtfz = gaa * az;
    rbmd::Real dtgx = fga * ax - hgb * bx;
    rbmd::Real dtgy = fga * ay - hgb * by;
    rbmd::Real dtgz = fga * az - hgb * bz;
    rbmd::Real dthx = gbb * bx;
    rbmd::Real dthy = gbb * by;
    rbmd::Real dthz = gbb * bz;

    rbmd::Real sx2 = df * dtgx;
    rbmd::Real sy2 = df * dtgy;
    rbmd::Real sz2 = df * dtgz;

    rbmd::Real f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, f4x, f4y, f4z;

    f1x = df * dtfx;
    f1y = df * dtfy;
    f1z = df * dtfz;

    f4x = df * dthx;
    f4y = df * dthy;
    f4z = df * dthz;

    f2x = sx2 - f1x;
    f2y = sy2 - f1y;
    f2z = sz2 - f1z;

    f3x = -sx2 - f4x;
    f3y = -sy2 - f4y;
    f3z = -sz2 - f4z;

    // 5. Atomically add forces
    atomicAdd(&fx[i1], f1x);
    atomicAdd(&fy[i1], f1y);
    atomicAdd(&fz[i1], f1z);

    atomicAdd(&fx[i2], f2x);
    atomicAdd(&fy[i2], f2y);
    atomicAdd(&fz[i2], f2z);

    atomicAdd(&fx[i3], f3x);
    atomicAdd(&fy[i3], f3y);
    atomicAdd(&fz[i3], f3z);

    atomicAdd(&fx[i4], f4x);
    atomicAdd(&fy[i4], f4y);
    atomicAdd(&fz[i4], f4z);

    // 6. Virial calculation (following your established pattern)
    rbmd::Real global_virial_temp[6];
    global_virial_temp[0] = (vb1x * f1x + vb2x * f3x + (vb3x + vb2x) * f4x);
    global_virial_temp[1] = (vb1y * f1y + vb2y * f3y + (vb3y + vb2y) * f4y);
    global_virial_temp[2] = (vb1z * f1z + vb2z * f3z + (vb3z + vb2z) * f4z);
    global_virial_temp[3] = (vb1x * f1y + vb2x * f3y + (vb3x + vb2x) * f4y);
    global_virial_temp[4] = (vb1x * f1z + vb2x * f3z + (vb3x + vb2x) * f4z);
    global_virial_temp[5] = (vb1y * f1z + vb2y * f3z + (vb3y + vb2y) * f4z);

    global_virial[0 * num_dihedrals + tid1] = global_virial_temp[0];
    global_virial[1 * num_dihedrals + tid1] = global_virial_temp[1];
    global_virial[2 * num_dihedrals + tid1] = global_virial_temp[2];
    global_virial[3 * num_dihedrals + tid1] = global_virial_temp[3];
    global_virial[4 * num_dihedrals + tid1] = global_virial_temp[4];
    global_virial[5 * num_dihedrals + tid1] = global_virial_temp[5];

    rbmd::Real local_virial[6];
    local_virial[0] = 0.25 * global_virial_temp[0];
    local_virial[1] = 0.25 * global_virial_temp[1];
    local_virial[2] = 0.25 * global_virial_temp[2];
    local_virial[3] = 0.25 * global_virial_temp[3];
    local_virial[4] = 0.25 * global_virial_temp[4];
    local_virial[5] = 0.25 * global_virial_temp[5];

    atomicAdd(&flat_virial[0 * num_atoms + i1], local_virial[0]);
    atomicAdd(&flat_virial[1 * num_atoms + i1], local_virial[1]);
    atomicAdd(&flat_virial[2 * num_atoms + i1], local_virial[2]);
    atomicAdd(&flat_virial[3 * num_atoms + i1], local_virial[3]);
    atomicAdd(&flat_virial[4 * num_atoms + i1], local_virial[4]);
    atomicAdd(&flat_virial[5 * num_atoms + i1], local_virial[5]);

    // (add for atoms i2, i3, i4 similarly)
    atomicAdd(&flat_virial[0 * num_atoms + i2], local_virial[0]);
    atomicAdd(&flat_virial[1 * num_atoms + i2], local_virial[1]);
    atomicAdd(&flat_virial[2 * num_atoms + i2], local_virial[2]);
    atomicAdd(&flat_virial[3 * num_atoms + i2], local_virial[3]);
    atomicAdd(&flat_virial[4 * num_atoms + i2], local_virial[4]);
    atomicAdd(&flat_virial[5 * num_atoms + i2], local_virial[5]);

    atomicAdd(&flat_virial[0 * num_atoms + i3], local_virial[0]);
    atomicAdd(&flat_virial[1 * num_atoms + i3], local_virial[1]);
    atomicAdd(&flat_virial[2 * num_atoms + i3], local_virial[2]);
    atomicAdd(&flat_virial[3 * num_atoms + i3], local_virial[3]);
    atomicAdd(&flat_virial[4 * num_atoms + i3], local_virial[4]);
    atomicAdd(&flat_virial[5 * num_atoms + i3], local_virial[5]);

    atomicAdd(&flat_virial[0 * num_atoms + i4], local_virial[0]);
    atomicAdd(&flat_virial[1 * num_atoms + i4], local_virial[1]);
    atomicAdd(&flat_virial[2 * num_atoms + i4], local_virial[2]);
    atomicAdd(&flat_virial[3 * num_atoms + i4], local_virial[3]);
    atomicAdd(&flat_virial[4 * num_atoms + i4], local_virial[4]);
    atomicAdd(&flat_virial[5 * num_atoms + i4], local_virial[5]);
  }

  // 7. Reduce energy across the block
  rbmd::Real block_sum =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage).Sum(local_energy_dihedral);

  if (threadIdx.x == 0) {
    atomicAdd(energy_dihedral, block_sum);
  }
}

__global__ void ComputeDihedralFourierForcePerAtom(
    Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
    const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
    const rbmd::Id* nterms, const rbmd::Id* fourier_offsets,
    const rbmd::Real* fourier_k, const rbmd::Id* fourier_multiplicity,
    const rbmd::Real* fourier_cos_shift, const rbmd::Real* fourier_sin_shift,
    const int* d_num_dihedral, const int* d_dihedral_type,
    const rbmd::Id* d_dihedral_atom1, const rbmd::Id* d_dihedral_atom2,
    const rbmd::Id* d_dihedral_atom3, const rbmd::Id* d_dihedral_atom4,
    int dihedral_per_atom, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
    rbmd::Real* flat_virial, rbmd::Real* energy_dihedral,
    unsigned long long* count_dihedral) {
  __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
      temp_storage_energy;
  __shared__ typename BLOCKREDUCE<unsigned long long, BLOCK_SIZE>::TempStorage
      temp_storage_count;

  rbmd::Real local_energy_dihedral = 0.0;
  unsigned long long local_count_dihedral = 0ULL;

  const unsigned int self_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (self_idx < static_cast<unsigned int>(num_local_atoms)) {
    const rbmd::Id self_gid = atoms_id[self_idx];
    rbmd::Real sum_fx = 0.0;
    rbmd::Real sum_fy = 0.0;
    rbmd::Real sum_fz = 0.0;
    rbmd::Real sum_v0 = 0.0;
    rbmd::Real sum_v1 = 0.0;
    rbmd::Real sum_v2 = 0.0;
    rbmd::Real sum_v3 = 0.0;
    rbmd::Real sum_v4 = 0.0;
    rbmd::Real sum_v5 = 0.0;

    int num_dihedral_self = d_num_dihedral[self_idx];
    if (num_dihedral_self < 0) {
      num_dihedral_self = 0;
    }
    if (num_dihedral_self > dihedral_per_atom) {
      num_dihedral_self = dihedral_per_atom;
    }

    for (int k = 0; k < num_dihedral_self; ++k) {
      const int topo_offset =
          static_cast<int>(self_idx) * dihedral_per_atom + k;
      const int type = d_dihedral_type[topo_offset];
      if (type < 0) {
        continue;
      }

      const rbmd::Id i1_id = d_dihedral_atom1[topo_offset];
      const rbmd::Id i2_id = d_dihedral_atom2[topo_offset];
      const rbmd::Id i3_id = d_dihedral_atom3[topo_offset];
      const rbmd::Id i4_id = d_dihedral_atom4[topo_offset];

      const rbmd::Id i1 = atom_id_to_idx[i1_id];
      const rbmd::Id i2 = atom_id_to_idx[i2_id];
      const rbmd::Id i3 = atom_id_to_idx[i3_id];
      const rbmd::Id i4 = atom_id_to_idx[i4_id];
      if (i1 < 0 || i2 < 0 || i3 < 0 || i4 < 0 || i1 >= num_total_atoms ||
          i2 >= num_total_atoms || i3 >= num_total_atoms ||
          i4 >= num_total_atoms) {
        continue;
      }

      rbmd::Real vb1x = px[i1] - px[i2];
      rbmd::Real vb1y = py[i1] - py[i2];
      rbmd::Real vb1z = pz[i1] - pz[i2];
      MinImageDistance_while(box, vb1x, vb1y, vb1z);

      rbmd::Real vb2x = px[i3] - px[i2];
      rbmd::Real vb2y = py[i3] - py[i2];
      rbmd::Real vb2z = pz[i3] - pz[i2];
      MinImageDistance_while(box, vb2x, vb2y, vb2z);

      rbmd::Real vb3x = px[i4] - px[i3];
      rbmd::Real vb3y = py[i4] - py[i3];
      rbmd::Real vb3z = pz[i4] - pz[i3];
      MinImageDistance_while(box, vb3x, vb3y, vb3z);

      rbmd::Real vb2xm = -vb2x;
      rbmd::Real vb2ym = -vb2y;
      rbmd::Real vb2zm = -vb2z;

      rbmd::Real ax = vb1y * vb2zm - vb1z * vb2ym;
      rbmd::Real ay = vb1z * vb2xm - vb1x * vb2zm;
      rbmd::Real az = vb1x * vb2ym - vb1y * vb2xm;
      rbmd::Real bx = vb3y * vb2zm - vb3z * vb2ym;
      rbmd::Real by = vb3z * vb2xm - vb3x * vb2zm;
      rbmd::Real bz = vb3x * vb2ym - vb3y * vb2xm;

      rbmd::Real rasq = ax * ax + ay * ay + az * az;
      rbmd::Real rbsq = bx * bx + by * by + bz * bz;
      rbmd::Real rgsq = vb2xm * vb2xm + vb2ym * vb2ym + vb2zm * vb2zm;
      rbmd::Real rg = SQRT(rgsq);

      rbmd::Real rginv = 0.0;
      rbmd::Real ra2inv = 0.0;
      rbmd::Real rb2inv = 0.0;
      if (rg > 0) rginv = 1.0 / rg;
      if (rasq > 0) ra2inv = 1.0 / rasq;
      if (rbsq > 0) rb2inv = 1.0 / rbsq;
      rbmd::Real rabinv = SQRT(ra2inv * rb2inv);

      rbmd::Real c = (ax * bx + ay * by + az * bz) * rabinv;
      rbmd::Real s = rg * rabinv * (ax * vb3x + ay * vb3y + az * vb3z);

      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;

      rbmd::Real edihedral = 0.0;
      rbmd::Real df = 0.0;

      rbmd::Id num_terms_for_type = nterms[type];
      rbmd::Id offset = fourier_offsets[type];

      for (rbmd::Id term = 0; term < num_terms_for_type; ++term) {
        rbmd::Id term_idx = offset + term;
        rbmd::Real k_j = fourier_k[term_idx];
        rbmd::Id m_j = fourier_multiplicity[term_idx];
        rbmd::Real cos_shift_j = fourier_cos_shift[term_idx];
        rbmd::Real sin_shift_j = fourier_sin_shift[term_idx];

        rbmd::Real p_ = 1.0;
        rbmd::Real df1_ = 0.0;
        rbmd::Real ddf1_ = 0.0;

        for (rbmd::Id iter = 0; iter < m_j; ++iter) {
          ddf1_ = p_ * c - df1_ * s;
          df1_ = p_ * s + df1_ * c;
          p_ = ddf1_;
        }

        rbmd::Real p_shifted = p_ * cos_shift_j + df1_ * sin_shift_j;
        rbmd::Real df1_shifted = df1_ * cos_shift_j - p_ * sin_shift_j;
        df1_shifted *= m_j;

        if (m_j == 0) {
          p_shifted = cos_shift_j;
          df1_shifted = 0.0;
        }

        edihedral += k_j * (1.0 + p_shifted);
        df += k_j * df1_shifted;
      }

      df = -df;

      const rbmd::Id min_gid = MIN(MIN(i1_id, i2_id), MIN(i3_id, i4_id));
      if (self_gid == min_gid) {
        local_energy_dihedral += edihedral;
        local_count_dihedral += 1ULL;
      }

      rbmd::Real fg = vb1x * vb2xm + vb1y * vb2ym + vb1z * vb2zm;
      rbmd::Real hg = vb3x * vb2xm + vb3y * vb2ym + vb3z * vb2zm;

      rbmd::Real fga = fg * ra2inv * rginv;
      rbmd::Real hgb = hg * rb2inv * rginv;
      rbmd::Real gaa = -ra2inv * rg;
      rbmd::Real gbb = rb2inv * rg;

      rbmd::Real dtfx = gaa * ax;
      rbmd::Real dtfy = gaa * ay;
      rbmd::Real dtfz = gaa * az;
      rbmd::Real dtgx = fga * ax - hgb * bx;
      rbmd::Real dtgy = fga * ay - hgb * by;
      rbmd::Real dtgz = fga * az - hgb * bz;
      rbmd::Real dthx = gbb * bx;
      rbmd::Real dthy = gbb * by;
      rbmd::Real dthz = gbb * bz;

      rbmd::Real sx2 = df * dtgx;
      rbmd::Real sy2 = df * dtgy;
      rbmd::Real sz2 = df * dtgz;

      rbmd::Real f1x = df * dtfx;
      rbmd::Real f1y = df * dtfy;
      rbmd::Real f1z = df * dtfz;

      rbmd::Real f4x = df * dthx;
      rbmd::Real f4y = df * dthy;
      rbmd::Real f4z = df * dthz;

      rbmd::Real f2x = sx2 - f1x;
      rbmd::Real f2y = sy2 - f1y;
      rbmd::Real f2z = sz2 - f1z;

      rbmd::Real f3x = -sx2 - f4x;
      rbmd::Real f3y = -sy2 - f4y;
      rbmd::Real f3z = -sz2 - f4z;

      rbmd::Real global_virial_temp[6];
      global_virial_temp[0] = (vb1x * f1x + vb2x * f3x + (vb3x + vb2x) * f4x);
      global_virial_temp[1] = (vb1y * f1y + vb2y * f3y + (vb3y + vb2y) * f4y);
      global_virial_temp[2] = (vb1z * f1z + vb2z * f3z + (vb3z + vb2z) * f4z);
      global_virial_temp[3] = (vb1x * f1y + vb2x * f3y + (vb3x + vb2x) * f4y);
      global_virial_temp[4] = (vb1x * f1z + vb2x * f3z + (vb3x + vb2x) * f4z);
      global_virial_temp[5] = (vb1y * f1z + vb2y * f3z + (vb3y + vb2y) * f4z);

      rbmd::Real local_virial[6];
      local_virial[0] = 0.25 * global_virial_temp[0];
      local_virial[1] = 0.25 * global_virial_temp[1];
      local_virial[2] = 0.25 * global_virial_temp[2];
      local_virial[3] = 0.25 * global_virial_temp[3];
      local_virial[4] = 0.25 * global_virial_temp[4];
      local_virial[5] = 0.25 * global_virial_temp[5];

      if (self_gid == i1_id) {
        sum_fx += f1x;
        sum_fy += f1y;
        sum_fz += f1z;
        sum_v0 += local_virial[0];
        sum_v1 += local_virial[1];
        sum_v2 += local_virial[2];
        sum_v3 += local_virial[3];
        sum_v4 += local_virial[4];
        sum_v5 += local_virial[5];
      } else if (self_gid == i2_id) {
        sum_fx += f2x;
        sum_fy += f2y;
        sum_fz += f2z;
        sum_v0 += local_virial[0];
        sum_v1 += local_virial[1];
        sum_v2 += local_virial[2];
        sum_v3 += local_virial[3];
        sum_v4 += local_virial[4];
        sum_v5 += local_virial[5];
      } else if (self_gid == i3_id) {
        sum_fx += f3x;
        sum_fy += f3y;
        sum_fz += f3z;
        sum_v0 += local_virial[0];
        sum_v1 += local_virial[1];
        sum_v2 += local_virial[2];
        sum_v3 += local_virial[3];
        sum_v4 += local_virial[4];
        sum_v5 += local_virial[5];
      } else if (self_gid == i4_id) {
        sum_fx += f4x;
        sum_fy += f4y;
        sum_fz += f4z;
        sum_v0 += local_virial[0];
        sum_v1 += local_virial[1];
        sum_v2 += local_virial[2];
        sum_v3 += local_virial[3];
        sum_v4 += local_virial[4];
        sum_v5 += local_virial[5];
      }
    }

    fx[self_idx] = sum_fx;
    fy[self_idx] = sum_fy;
    fz[self_idx] = sum_fz;
    flat_virial[0 * num_local_atoms + self_idx] = sum_v0;
    flat_virial[1 * num_local_atoms + self_idx] = sum_v1;
    flat_virial[2 * num_local_atoms + self_idx] = sum_v2;
    flat_virial[3 * num_local_atoms + self_idx] = sum_v3;
    flat_virial[4 * num_local_atoms + self_idx] = sum_v4;
    flat_virial[5 * num_local_atoms + self_idx] = sum_v5;
  }

  const rbmd::Real block_sum_energy =
      BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_energy)
          .Sum(local_energy_dihedral);
  const unsigned long long block_sum_count =
      BLOCKREDUCE<unsigned long long, BLOCK_SIZE>(temp_storage_count)
          .Sum(local_count_dihedral);

  if (threadIdx.x == 0) {
    atomicAdd(energy_dihedral, block_sum_energy);
    atomicAdd(count_dihedral, block_sum_count);
  }
}

  //Imprope
  __global__ void ComputeImproperHarmonicForce(
    Box box,const rbmd::Id num_atoms,const rbmd::Id num_impropers,
    const rbmd::Id* atom_id_to_idx,const rbmd::Real* improper_coeffs_k,
    const rbmd::Real* improper_coeffs_chi,const rbmd::Id* improper_type,
    const rbmd::Id* improperlisti,const rbmd::Id* improperlistj,
    const rbmd::Id* improperlistk,const rbmd::Id* improperlistw,
    const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
    rbmd::Real* fx,rbmd::Real* fy,rbmd::Real* fz,
    rbmd::Real* flat_virial,rbmd::Real* energy_improper) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage;
    rbmd::Real local_energy_improper = 0;

    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_impropers) {
      rbmd::Id improperi = improperlisti[tid1];
      rbmd::Id improperj = improperlistj[tid1];
      rbmd::Id improperk = improperlistk[tid1];
      rbmd::Id improperw = improperlistw[tid1];

      rbmd::Id improperii = atom_id_to_idx[improperi];
      rbmd::Id improperjj = atom_id_to_idx[improperj];
      rbmd::Id improperkk = atom_id_to_idx[improperk];
      rbmd::Id improperww = atom_id_to_idx[improperw];
      rbmd::Id impropertype = improper_type[tid1];

      rbmd::Real k = improper_coeffs_k[impropertype];
      rbmd::Real chi = improper_coeffs_chi[impropertype];
      rbmd::Real x12 = px[improperii] - px[improperjj];  // i j =vb1
      rbmd::Real y12 = py[improperii] - py[improperjj];
      rbmd::Real z12 = pz[improperii] - pz[improperjj];
      MinImageDistance_while(box, x12, y12, z12);

      rbmd::Real x23 = px[improperkk] - px[improperjj];  //  k j=vb2
      rbmd::Real y23 = py[improperkk] - py[improperjj];
      rbmd::Real z23 = pz[improperkk] - pz[improperjj];
      MinImageDistance_while(box, x23, y23, z23);

      rbmd::Real x34 = px[improperww] - px[improperkk];  // w k   =vb3
      rbmd::Real y34 = py[improperww] - py[improperkk];
      rbmd::Real z34 = pz[improperww] - pz[improperkk];
      MinImageDistance_while(box, x34, y34, z34);

      rbmd::Real ss1 = 1.0 / (x12 * x12 + y12 * y12 + z12 * z12);
      rbmd::Real ss2 = 1.0 / (x23 * x23 + y23 * y23 + z23 * z23);
      rbmd::Real ss3 = 1.0 / (x34 * x34 + y34 * y34 + z34 * z34);

      rbmd::Real r1 = SQRT(ss1);
      rbmd::Real r2 = SQRT(ss2);
      rbmd::Real r3 = SQRT(ss3);

      // sin and cos of angle

      rbmd::Real c0 = (x12 * x34 + y12 * y34 + z12 * z34) * r1 * r3;
      rbmd::Real c1 = (x12 * x23 + y12 * y23 + z12 * z23) * r1 * r2;
      rbmd::Real c2 = -(x34 * x23 + y34 * y23 + z34 * z23) * r3 * r2;

      rbmd::Real  s1 = 1.0 - c1 * c1;
      if (s1 < SMALL) s1 = SMALL;
      s1 = 1.0 / s1;

       rbmd::Real s2 = 1.0 - c2 * c2;
      if (s2 < SMALL) s2 = SMALL;
      s2 = 1.0 / s2;

      rbmd::Real s12 = SQRT(s1 * s2);
      rbmd::Real c = (c1 * c2 + c0) * s12;

      // error check

      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;

      rbmd::Real s = SQRT(1.0 - c * c);
      if (s < SMALL) s = SMALL;


      // force & energy
      rbmd::Real domega = ACOS(c) - chi;
      rbmd::Real a = k * domega;
      local_energy_improper = a * domega;

      a = -a * 2.0 / s;
      c = c * a;
      s12 = s12 * a;
      rbmd::Real a11 = c * ss1 * s1;
      rbmd::Real a22 = -ss2 * (2.0 * c0 * s12 - c * (s1 + s2));
      rbmd::Real a33 = c * ss3 * s2;
      rbmd::Real a12 = -r1 * r2 * (c1 * c * s1 + c2 * s12);
      rbmd::Real a13 = -r1 * r3 * s12;
      rbmd::Real a23 = r2 * r3 * (c2 * c * s2 + c1 * s12);

      rbmd::Real sx2 = a22 * x23 + a23 * x34 + a12 * x12;
      rbmd::Real sy2 = a22 * y23 + a23 * y34 + a12 * y12;
      rbmd::Real sz2 = a22 * z23 + a23 * z34 + a12 * z12;

      // force
      rbmd::Real force_improperi_x, force_improperi_y, force_improperi_z;
      rbmd::Real force_improperj_x, force_improperj_y, force_improperj_z;
      rbmd::Real force_improperk_x, force_improperk_y, force_improperk_z;
      rbmd::Real force_improperw_x, force_improperw_y, force_improperw_z;

      force_improperi_x = a12 * x23 + a13 * x34 + a11 * x12;
      force_improperi_y = a12 * y23 + a13 * y34 + a11 * y12;
      force_improperi_z = a12 * z23 + a13 * z34 + a11 * z12;

      force_improperj_x = -sx2 - force_improperi_x;
      force_improperj_y = -sy2 - force_improperi_y;
      force_improperj_z = -sz2 - force_improperi_z;

      force_improperw_x = a23 * x23 + a33 * x34 + a13 * x12;
      force_improperw_y = a23 * y23 + a33 * y34 + a13 * y12;
      force_improperw_z = a23 * z23 + a33 * z34 + a13 * z12;

      force_improperk_x = sx2 - force_improperw_x;
      force_improperk_y = sy2 - force_improperw_y;
      force_improperk_z = sz2 - force_improperw_z;


      atomicAdd(&fx[improperii], force_improperi_x);
      atomicAdd(&fy[improperii], force_improperi_y);
      atomicAdd(&fz[improperii], force_improperi_z);

      atomicAdd(&fx[improperjj], force_improperj_x);
      atomicAdd(&fy[improperjj], force_improperj_y);
      atomicAdd(&fz[improperjj], force_improperj_z);

      atomicAdd(&fx[improperww], force_improperw_x);
      atomicAdd(&fy[improperww], force_improperw_y);
      atomicAdd(&fz[improperww], force_improperw_z);

      atomicAdd(&fx[improperkk], force_improperk_x);
      atomicAdd(&fy[improperkk], force_improperk_y);
      atomicAdd(&fz[improperkk], force_improperk_z);

      //
      rbmd::Real global_virial_temp[6];
      global_virial_temp[0] = (x12 * force_improperi_x + x23 * force_improperk_x +
          (x34 + x23) * force_improperw_x);

      global_virial_temp[1] = (y12 * force_improperi_y+ y23 * force_improperk_y +
         (y34 + y23) * force_improperw_y);

      global_virial_temp[2] = (z12 * force_improperi_z + z23 * force_improperk_z +
         (z34 + z23) * force_improperw_z);

      global_virial_temp[3] = (x12 * force_improperi_y + x23 * force_improperk_y +
         (x34 + x23) * force_improperw_y);

      global_virial_temp[4] = (x12* force_improperi_z + x23 * force_improperk_z +
         (x34 + x23) * force_improperw_z);

      global_virial_temp[5] =  (y12 * force_improperi_z + y23 * force_improperk_z +
         (y34 + y23) * force_improperw_z);

      rbmd::Real local_virial[6];
      local_virial[0] = 0.25* global_virial_temp[0];
      local_virial[1] = 0.25* global_virial_temp[1];
      local_virial[2] = 0.25* global_virial_temp[2];
      local_virial[3] = 0.25* global_virial_temp[3];
      local_virial[4] = 0.25* global_virial_temp[4];
      local_virial[5] = 0.25* global_virial_temp[5];

      //
      atomicAdd(&flat_virial[0 * num_atoms + improperii], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperii], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperii], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperii], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperii], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperii], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + improperjj], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperjj], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperjj], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperjj], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperjj], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperjj], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + improperkk], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperkk], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperkk], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperkk], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperkk], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperkk], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + improperww], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperww], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperww], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperww], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperww], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperww], local_virial[5]);
    }
    rbmd::Real block_sum =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
            .Sum(local_energy_improper);

    if (threadIdx.x == 0) {
      atomicAdd(energy_improper, block_sum);
    }
  }

  __global__ void ComputeImproperCVFFForce(
    Box box,const rbmd::Id num_atoms,const rbmd::Id num_impropers,
    const rbmd::Id* atom_id_to_idx,const rbmd::Real* improper_coeffs_k,
    const rbmd::Id* improper_coeffs_d,const rbmd::Id* improper_coeffs_n,
    const rbmd::Id* improper_type,
    const rbmd::Id* improperlisti,const rbmd::Id* improperlistj,
    const rbmd::Id* improperlistk,const rbmd::Id* improperlistw,
    const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
    rbmd::Real* fx,rbmd::Real* fy,rbmd::Real* fz,
    rbmd::Real* flat_virial,rbmd::Real* energy_improper) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage;
    rbmd::Real local_energy_improper = 0;

    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_impropers) {
      rbmd::Id improperi = improperlisti[tid1];
      rbmd::Id improperj = improperlistj[tid1];
      rbmd::Id improperk = improperlistk[tid1];
      rbmd::Id improperw = improperlistw[tid1];

      rbmd::Id improperii = atom_id_to_idx[improperi];
      rbmd::Id improperjj = atom_id_to_idx[improperj];
      rbmd::Id improperkk = atom_id_to_idx[improperk];
      rbmd::Id improperww = atom_id_to_idx[improperw];
      rbmd::Id impropertype = improper_type[tid1];

      rbmd::Real k = improper_coeffs_k[impropertype];
      rbmd::Id d = improper_coeffs_d[impropertype]; // -1  or  1
      rbmd::Id n = improper_coeffs_n[impropertype]; // 0 1 2 3 4 5

      rbmd::Real x12 = px[improperii] - px[improperjj];  // i j =vb1
      rbmd::Real y12 = py[improperii] - py[improperjj];
      rbmd::Real z12 = pz[improperii] - pz[improperjj];
      MinImageDistance_while(box, x12, y12, z12);

      rbmd::Real x23 = px[improperkk] - px[improperjj];  //  k j=vb2
      rbmd::Real y23 = py[improperkk] - py[improperjj];
      rbmd::Real z23 = pz[improperkk] - pz[improperjj];
      MinImageDistance_while(box, x23, y23, z23);

      rbmd::Real  x23m = -x23;
      rbmd::Real  y23m = -y23;
      rbmd::Real  z23m = -z23;

      rbmd::Real x34 = px[improperww] - px[improperkk];  // w k   =vb3
      rbmd::Real y34 = py[improperww] - py[improperkk];
      rbmd::Real z34 = pz[improperww] - pz[improperkk];
      MinImageDistance_while(box, x34, y34, z34);

      rbmd::Real sb1 = 1.0 / (x12 * x12 + y12 * y12 + z12 * z12);
      rbmd::Real sb2 = 1.0 / (x23 * x23 + y23 * y23 + z23 * z23);
      rbmd::Real sb3 = 1.0 / (x34 * x34 + y34 * y34 + z34 * z34);

      rbmd::Real rb1 = SQRT(sb1);
      rbmd::Real rb3 = SQRT(sb3);

      rbmd::Real c0 = (x12 * x34 + y12 * y34 + z12 * z34) * rb1 * rb3;
      // 1st and 2nd angle
      rbmd::Real  b1mag2, b1mag, b2mag2;
      rbmd::Real b2mag, b3mag2, b3mag, ctmp, r12c1, c1mag, r12c2;
      rbmd::Real c2mag, sc1, sc2, s1, s2, s12, c, p, pd, rc2, a, a11, a22;
      rbmd::Real a33, a12, a13, a23, sx2, sy2, sz2;


      b1mag2 = x12 * x12 + y12 * y12 + z12 * z12;
      b1mag = SQRT(b1mag2);
      b2mag2 = x23 * x23 + y23 * y23 + z23 * z23;
      b2mag = SQRT(b2mag2);
      b3mag2 =x34 * x34 + y34 * y34 + z34 * z34;
      b3mag = SQRT(b3mag2);

      ctmp = x12 * x23 + y12 * y23 + z12 * z23;
      r12c1 = 1.0 / (b1mag * b2mag);
      c1mag = ctmp * r12c1;

      ctmp = x23m * x34 + y23m * y34 + z23m * z34;
      r12c2 = 1.0 / (b2mag * b3mag);
      c2mag = ctmp * r12c2;
      // cos and sin of 2 angles and final c

      sc1 = SQRT(1.0 - c1mag * c1mag);
      if (sc1 < SMALL) sc1 = SMALL;
      sc1 = 1.0 / sc1;

      sc2 = SQRT(1.0 - c2mag * c2mag);
      if (sc2 < SMALL) sc2 = SMALL;
      sc2 = 1.0 / sc2;

      s1 = sc1 * sc1;
      s2 = sc2 * sc2;
      s12 = sc1 * sc2;
      c = (c0 + c1mag * c2mag) * s12;

      // error check


      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;

      rbmd::Id m = n ;

      if (m == 2) {
        p = 2.0 * c * c;
        pd = 2.0 * c;
      } else if (m == 3) {
        rc2 = c * c;
        p = (4.0 * rc2 - 3.0) * c + 1.0;
        pd = 6.0 * rc2 - 1.5;
      } else if (m == 4) {
        rc2 = c * c;
        p = 8.0 * (rc2 - 1) * rc2 + 2.0;
        pd = (16.0 * rc2 - 8.0) * c;
      } else if (m == 6) {
        rc2 = c * c;
        p = ((32.0 * rc2 - 48.0) * rc2 + 18.0) * rc2;
        pd = (96.0 * (rc2 - 1.0) * rc2 + 18.0) * c;
      } else if (m == 1) {
        p = c + 1.0;
        pd = 0.5;
      } else if (m == 5) {
        rc2 = c * c;
        p = ((16.0 * rc2 - 20.0) * rc2 + 5.0) * c + 1.0;
        pd = (40.0 * rc2 - 30.0) * rc2 + 2.5;
      } else if (m == 0) {
        p = 2.0;
        pd = 0.0;
      }

      if (d == -1) {
        p = 2.0 - p;
        pd = -pd;
      }

      local_energy_improper = k * p;
      //printf("local_energy_improper:  %f\n ",local_energy_improper);

      a = 2.0 * k * pd;
      c = c * a;
      s12 = s12 * a;
      a11 = c * sb1 * s1;
      a22 = -sb2 * (2.0 * c0 * s12 - c * (s1 + s2));
      a33 = c * sb3 * s2;
      a12 = -r12c1 * (c1mag * c * s1 + c2mag * s12);
      a13 = -rb1 * rb3 * s12;
      a23 = r12c2 * (c2mag * c * s2 + c1mag * s12);

      sx2 = a12 * x12 + a22 * x23 + a23 * x34;
      sy2 = a12 * y12 + a22 * y23 + a23 * y34;
      sz2 = a12 * z12 + a22 * z23 + a23 * z34;

      // force
      rbmd::Real force_improperi_x, force_improperi_y, force_improperi_z;
      rbmd::Real force_improperj_x, force_improperj_y, force_improperj_z;
      rbmd::Real force_improperk_x, force_improperk_y, force_improperk_z;
      rbmd::Real force_improperw_x, force_improperw_y, force_improperw_z;

      force_improperi_x = a11 * x12 + a12 * x23 + a13 * x34;
      force_improperi_y = a11 * y12 + a12 * y23 + a13 * y34;
      force_improperi_z = a11 * z12 + a12 * z23 + a13 * z34;

      force_improperj_x = -sx2 - force_improperi_x;
      force_improperj_y = -sy2 - force_improperi_y;
      force_improperj_z = -sz2 - force_improperi_z;

      force_improperw_x = a13 * x12 + a23 * x23 + a33 * x34;
      force_improperw_y = a13 * y12 + a23 * y23 + a33 * y34;
      force_improperw_z = a13 * z12 + a23 * z23 + a33 * z34;

      force_improperk_x = sx2 - force_improperw_x;
      force_improperk_y = sy2 - force_improperw_y;
      force_improperk_z = sz2 - force_improperw_z;


      atomicAdd(&fx[improperii], force_improperi_x);
      atomicAdd(&fy[improperii], force_improperi_y);
      atomicAdd(&fz[improperii], force_improperi_z);

      atomicAdd(&fx[improperjj], force_improperj_x);
      atomicAdd(&fy[improperjj], force_improperj_y);
      atomicAdd(&fz[improperjj], force_improperj_z);

      atomicAdd(&fx[improperww], force_improperw_x);
      atomicAdd(&fy[improperww], force_improperw_y);
      atomicAdd(&fz[improperww], force_improperw_z);

      atomicAdd(&fx[improperkk], force_improperk_x);
      atomicAdd(&fy[improperkk], force_improperk_y);
      atomicAdd(&fz[improperkk], force_improperk_z);

      //
      rbmd::Real global_virial_temp[6];
      global_virial_temp[0] = (x12 * force_improperi_x + x23 * force_improperk_x +
          (x34 + x23) * force_improperw_x);

      global_virial_temp[1] = (y12 * force_improperi_y+ y23 * force_improperk_y +
         (y34 + y23) * force_improperw_y);

      global_virial_temp[2] = (z12 * force_improperi_z + z23 * force_improperk_z +
         (z34 + z23) * force_improperw_z);

      global_virial_temp[3] = (x12 * force_improperi_y + x23 * force_improperk_y +
         (x34 + x23) * force_improperw_y);

      global_virial_temp[4] = (x12* force_improperi_z + x23 * force_improperk_z +
         (x34 + x23) * force_improperw_z);

      global_virial_temp[5] =  (y12 * force_improperi_z + y23 * force_improperk_z +
         (y34 + y23) * force_improperw_z);

      rbmd::Real local_virial[6];
      local_virial[0] = 0.25* global_virial_temp[0];
      local_virial[1] = 0.25* global_virial_temp[1];
      local_virial[2] = 0.25* global_virial_temp[2];
      local_virial[3] = 0.25* global_virial_temp[3];
      local_virial[4] = 0.25* global_virial_temp[4];
      local_virial[5] = 0.25* global_virial_temp[5];

      //
      atomicAdd(&flat_virial[0 * num_atoms + improperii], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperii], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperii], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperii], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperii], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperii], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + improperjj], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperjj], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperjj], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperjj], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperjj], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperjj], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + improperkk], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperkk], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperkk], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperkk], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperkk], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperkk], local_virial[5]);

      atomicAdd(&flat_virial[0 * num_atoms + improperww], local_virial[0]);
      atomicAdd(&flat_virial[1 * num_atoms + improperww], local_virial[1]);
      atomicAdd(&flat_virial[2 * num_atoms + improperww], local_virial[2]);
      atomicAdd(&flat_virial[3 * num_atoms + improperww], local_virial[3]);
      atomicAdd(&flat_virial[4 * num_atoms + improperww], local_virial[4]);
      atomicAdd(&flat_virial[5 * num_atoms + improperww], local_virial[5]);
    }
    rbmd::Real block_sum =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage)
            .Sum(local_energy_improper);

    if (threadIdx.x == 0) {
      atomicAdd(energy_improper, block_sum);
    }
  }

  __global__ void ComputeBondForcePerAtom(
      Box box, const rbmd::Id num_local_atoms,
      const rbmd::Id num_total_atoms,
      const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
      const rbmd::Id atom_id_to_idx_size,
      const rbmd::Real* bond_coeffs_k, int num_bond_types,
      const rbmd::Real* bond_coeffs_equilibrium,
      const int* d_num_bond, const int* d_bond_type,
      const rbmd::Id* d_bond_atom, int bond_per_atom,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* energy_bond, unsigned long long* count_bond) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_energy;
    __shared__ typename BLOCKREDUCE<unsigned long long, BLOCK_SIZE>::TempStorage
        temp_storage_count;

    rbmd::Real local_energy_bond = 0.0;
    unsigned long long local_count_bond = 0ULL;

    const unsigned int self_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (self_idx < static_cast<unsigned int>(num_local_atoms)) {
      const rbmd::Id self_gid = atoms_id[self_idx];
      rbmd::Real sum_fx = 0.0;
      rbmd::Real sum_fy = 0.0;
      rbmd::Real sum_fz = 0.0;

      int num_bond_self = d_num_bond[self_idx];
      if (num_bond_self < 0) {
        num_bond_self = 0;
      }
      if (num_bond_self > bond_per_atom) {
        num_bond_self = bond_per_atom;
      }

      for (int k = 0; k < num_bond_self; ++k) {
        const int topo_offset = static_cast<int>(self_idx) * bond_per_atom + k;
        const int bondtype = d_bond_type[topo_offset];
        const rbmd::Id other_gid = d_bond_atom[topo_offset];
        if (other_gid < 0 || other_gid >= atom_id_to_idx_size) {
          continue;
        }
        if (bondtype < 0 || bondtype >= num_bond_types) {
          continue;
        }

        const rbmd::Id other_idx = atom_id_to_idx[other_gid];
        if (other_idx < 0 || other_idx >= num_total_atoms) {
          continue;
        }
        if (atoms_id[other_idx] != other_gid) {
          continue;
        }

        const rbmd::Real k_coeff = bond_coeffs_k[bondtype];
        const rbmd::Real equilibrium_bond = bond_coeffs_equilibrium[bondtype];

        rbmd::Real x12 = px[self_idx] - px[other_idx];
        rbmd::Real y12 = py[self_idx] - py[other_idx];
        rbmd::Real z12 = pz[self_idx] - pz[other_idx];
        MinImageDistance_while(box, x12, y12, z12);

        const rbmd::Real dis_12 = SQRT(x12 * x12 + y12 * y12 + z12 * z12);
        const rbmd::Real dr = dis_12 - equilibrium_bond;
        const rbmd::Real rk = k_coeff * dr;

        rbmd::Real forcebond = 0.0;
        if (dis_12 > 0.01) {
          forcebond = -2.0 * rk / dis_12;
        }

        sum_fx += forcebond * x12;
        sum_fy += forcebond * y12;
        sum_fz += forcebond * z12;

        if (self_gid < other_gid) {
          local_energy_bond += rk * dr;
          local_count_bond += 1ULL;
        }
      }

      fx[self_idx] = sum_fx;
      fy[self_idx] = sum_fy;
      fz[self_idx] = sum_fz;
    }

    const rbmd::Real block_sum_energy =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_energy)
            .Sum(local_energy_bond);
    const unsigned long long block_sum_count =
        BLOCKREDUCE<unsigned long long, BLOCK_SIZE>(temp_storage_count)
            .Sum(local_count_bond);

    if (threadIdx.x == 0) {
      atomicAdd(energy_bond, block_sum_energy);
      atomicAdd(count_bond, block_sum_count);
    }
  }

  __global__ void ComputeAngleForcePerAtom(
      Box box, const rbmd::Id num_local_atoms,
      const rbmd::Id num_total_atoms,
      const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
      const rbmd::Real* anglel_coeffs_k,
      const rbmd::Real* anglel_coeffs_equilibrium, const int* d_num_angle,
      const int* d_angle_type, const rbmd::Id* d_angle_atom1,
      const rbmd::Id* d_angle_atom2, const rbmd::Id* d_angle_atom3,
      int angle_per_atom,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* energy_angle, unsigned long long* count_angle) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_energy;
    __shared__ typename BLOCKREDUCE<unsigned long long, BLOCK_SIZE>::TempStorage
        temp_storage_count;

    rbmd::Real local_energy_angle = 0.0;
    unsigned long long local_count_angle = 0ULL;

    const unsigned int self_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (self_idx < static_cast<unsigned int>(num_local_atoms)) {
      const rbmd::Id self_gid = atoms_id[self_idx];
      rbmd::Real sum_fx = 0.0;
      rbmd::Real sum_fy = 0.0;
      rbmd::Real sum_fz = 0.0;

      int num_angle_self = d_num_angle[self_idx];
      if (num_angle_self < 0) {
        num_angle_self = 0;
      }
      if (num_angle_self > angle_per_atom) {
        num_angle_self = angle_per_atom;
      }

      for (int k = 0; k < num_angle_self; ++k) {
        const int topo_offset = static_cast<int>(self_idx) * angle_per_atom + k;
        const int angletype = d_angle_type[topo_offset];
        if (angletype < 0) {
          continue;
        }

        const rbmd::Id gid_i = d_angle_atom1[topo_offset];
        const rbmd::Id gid_j = d_angle_atom2[topo_offset];
        const rbmd::Id gid_k = d_angle_atom3[topo_offset];

        const rbmd::Id idx_i = atom_id_to_idx[gid_i];
        const rbmd::Id idx_j = atom_id_to_idx[gid_j];
        const rbmd::Id idx_k = atom_id_to_idx[gid_k];
        if (idx_i < 0 || idx_j < 0 || idx_k < 0 ||
            idx_i >= num_total_atoms || idx_j >= num_total_atoms ||
            idx_k >= num_total_atoms) {
          continue;
        }

        const rbmd::Real k_coeff = anglel_coeffs_k[angletype];
        const rbmd::Real equilibrium_angle = anglel_coeffs_equilibrium[angletype];

        rbmd::Real x12 = px[idx_i] - px[idx_j];  // i - j (j is center)
        rbmd::Real y12 = py[idx_i] - py[idx_j];
        rbmd::Real z12 = pz[idx_i] - pz[idx_j];
        MinImageDistance_while(box, x12, y12, z12);

        rbmd::Real x23 = px[idx_k] - px[idx_j];  // k - j
        rbmd::Real y23 = py[idx_k] - py[idx_j];
        rbmd::Real z23 = pz[idx_k] - pz[idx_j];
        MinImageDistance_while(box, x23, y23, z23);

        const rbmd::Real dis_12_2 = x12 * x12 + y12 * y12 + z12 * z12;
        const rbmd::Real dis_12 = SQRT(dis_12_2);
        const rbmd::Real dis_23_2 = x23 * x23 + y23 * y23 + z23 * z23;
        const rbmd::Real dis_23 = SQRT(dis_23_2);

        rbmd::Real cosangle = x12 * x23 + y12 * y23 + z12 * z23;
        cosangle /= dis_12 * dis_23;
        if (cosangle > 1.0) cosangle = 1.0;
        if (cosangle < -1.0) cosangle = -1.0;

        rbmd::Real s = SQRT(1.0 - cosangle * cosangle);
        if (s < SMALL) s = SMALL;
        s = 1.0 / s;

        const rbmd::Real dtheta =
            ACOS(cosangle) - (equilibrium_angle * M_PI) / 180;
        const rbmd::Real tk = k_coeff * dtheta;

        const rbmd::Id min_gid = MIN(gid_i, MIN(gid_j, gid_k));
        if (self_gid == min_gid) {
          local_energy_angle += tk * dtheta;
          local_count_angle += 1ULL;
        }

        const rbmd::Real a = -2.0 * tk * s;
        const rbmd::Real a11 = a * cosangle / dis_12_2;
        const rbmd::Real a12 = -a / (dis_12 * dis_23);
        const rbmd::Real a22 = a * cosangle / dis_23_2;

        const rbmd::Real force_i_x = a11 * x12 + a12 * x23;
        const rbmd::Real force_i_y = a11 * y12 + a12 * y23;
        const rbmd::Real force_i_z = a11 * z12 + a12 * z23;

        const rbmd::Real force_k_x = a22 * x23 + a12 * x12;
        const rbmd::Real force_k_y = a22 * y23 + a12 * y12;
        const rbmd::Real force_k_z = a22 * z23 + a12 * z12;

        const rbmd::Real force_j_x = -(force_i_x + force_k_x);
        const rbmd::Real force_j_y = -(force_i_y + force_k_y);
        const rbmd::Real force_j_z = -(force_i_z + force_k_z);

        if (self_gid == gid_i) {
          sum_fx += force_i_x;
          sum_fy += force_i_y;
          sum_fz += force_i_z;
        } else if (self_gid == gid_j) {
          sum_fx += force_j_x;
          sum_fy += force_j_y;
          sum_fz += force_j_z;
        } else if (self_gid == gid_k) {
          sum_fx += force_k_x;
          sum_fy += force_k_y;
          sum_fz += force_k_z;
        }
      }

      fx[self_idx] = sum_fx;
      fy[self_idx] = sum_fy;
      fz[self_idx] = sum_fz;
    }

    const rbmd::Real block_sum_energy =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_energy)
            .Sum(local_energy_angle);
    const unsigned long long block_sum_count =
        BLOCKREDUCE<unsigned long long, BLOCK_SIZE>(temp_storage_count)
            .Sum(local_count_angle);

    if (threadIdx.x == 0) {
      atomicAdd(energy_angle, block_sum_energy);
      atomicAdd(count_angle, block_sum_count);
    }
  }

  __global__ void ComputeDihedralForcePerAtom(
      Box box, const rbmd::Id num_local_atoms,
      const rbmd::Id num_total_atoms,
      const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
      const rbmd::Real* dihedral_coeffs_k,
      const rbmd::Id* dihedral_coeffs_sign,
      const rbmd::Id* dihedral_coeffs_multiplicity, const int* d_num_dihedral,
      const int* d_dihedral_type, const rbmd::Id* d_dihedral_atom1,
      const rbmd::Id* d_dihedral_atom2, const rbmd::Id* d_dihedral_atom3,
      const rbmd::Id* d_dihedral_atom4, int dihedral_per_atom,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* flat_virial, rbmd::Real* energy_dihedral,
      unsigned long long* count_dihedral) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_energy;
    __shared__ typename BLOCKREDUCE<unsigned long long, BLOCK_SIZE>::TempStorage
        temp_storage_count;

    rbmd::Real local_energy_dihedral = 0.0;
    unsigned long long local_count_dihedral = 0ULL;

    const unsigned int self_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (self_idx < static_cast<unsigned int>(num_local_atoms)) {
      const rbmd::Id self_gid = atoms_id[self_idx];
      rbmd::Real sum_fx = 0.0;
      rbmd::Real sum_fy = 0.0;
      rbmd::Real sum_fz = 0.0;
      rbmd::Real sum_v0 = 0.0;
      rbmd::Real sum_v1 = 0.0;
      rbmd::Real sum_v2 = 0.0;
      rbmd::Real sum_v3 = 0.0;
      rbmd::Real sum_v4 = 0.0;
      rbmd::Real sum_v5 = 0.0;

      int num_dihedral_self = d_num_dihedral[self_idx];
      if (num_dihedral_self < 0) {
        num_dihedral_self = 0;
      }
      if (num_dihedral_self > dihedral_per_atom) {
        num_dihedral_self = dihedral_per_atom;
      }

      for (int k = 0; k < num_dihedral_self; ++k) {
        const int topo_offset =
            static_cast<int>(self_idx) * dihedral_per_atom + k;
        const int dihedraltype = d_dihedral_type[topo_offset];
        if (dihedraltype < 0) {
          continue;
        }

        const rbmd::Id gid_i = d_dihedral_atom1[topo_offset];
        const rbmd::Id gid_j = d_dihedral_atom2[topo_offset];
        const rbmd::Id gid_k = d_dihedral_atom3[topo_offset];
        const rbmd::Id gid_w = d_dihedral_atom4[topo_offset];

        const rbmd::Id idx_i = atom_id_to_idx[gid_i];
        const rbmd::Id idx_j = atom_id_to_idx[gid_j];
        const rbmd::Id idx_k = atom_id_to_idx[gid_k];
        const rbmd::Id idx_w = atom_id_to_idx[gid_w];
        if (idx_i < 0 || idx_j < 0 || idx_k < 0 || idx_w < 0 ||
            idx_i >= num_total_atoms || idx_j >= num_total_atoms ||
            idx_k >= num_total_atoms || idx_w >= num_total_atoms) {
          continue;
        }

        rbmd::Real cos_shift = -1.0;
        rbmd::Real sin_shift = 0.0;
        if (dihedral_coeffs_sign[dihedraltype] == 1) {
          cos_shift = 1.0;
        }

        const rbmd::Real k_coeff = dihedral_coeffs_k[dihedraltype];

        rbmd::Real x12 = px[idx_i] - px[idx_j];
        rbmd::Real y12 = py[idx_i] - py[idx_j];
        rbmd::Real z12 = pz[idx_i] - pz[idx_j];
        MinImageDistance_while(box, x12, y12, z12);

        rbmd::Real x23 = px[idx_k] - px[idx_j];
        rbmd::Real y23 = py[idx_k] - py[idx_j];
        rbmd::Real z23 = pz[idx_k] - pz[idx_j];
        MinImageDistance_while(box, x23, y23, z23);

        rbmd::Real x23m = -x23;
        rbmd::Real y23m = -y23;
        rbmd::Real z23m = -z23;

        rbmd::Real x34 = px[idx_w] - px[idx_k];
        rbmd::Real y34 = py[idx_w] - py[idx_k];
        rbmd::Real z34 = pz[idx_w] - pz[idx_k];
        MinImageDistance_while(box, x34, y34, z34);

        rbmd::Real ax = y12 * z23m - z12 * y23m;
        rbmd::Real ay = z12 * x23m - x12 * z23m;
        rbmd::Real az = x12 * y23m - y12 * x23m;
        rbmd::Real bx = y34 * z23m - z34 * y23m;
        rbmd::Real by = z34 * x23m - x34 * z23m;
        rbmd::Real bz = x34 * y23m - y34 * x23m;

        rbmd::Real rasq = ax * ax + ay * ay + az * az;
        rbmd::Real rbsq = bx * bx + by * by + bz * bz;
        rbmd::Real rgsq = x23m * x23m + y23m * y23m + z23m * z23m;
        rbmd::Real rg = SQRT(rgsq);

        rbmd::Real rginv = 0.0;
        rbmd::Real ra2inv = 0.0;
        rbmd::Real rb2inv = 0.0;
        if (rg > 0) rginv = 1.0 / rg;
        if (rasq > 0) ra2inv = 1.0 / rasq;
        if (rbsq > 0) rb2inv = 1.0 / rbsq;

        const rbmd::Real rabinv = SQRT(ra2inv * rb2inv);

        rbmd::Real c = (ax * bx + ay * by + az * bz) * rabinv;
        rbmd::Real s = rg * rabinv * (ax * x34 + ay * y34 + az * z34);
        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        const rbmd::Id m = dihedral_coeffs_multiplicity[dihedraltype];
        rbmd::Real p = 1.0;
        rbmd::Real ddf1 = 0.0;
        rbmd::Real df1 = 0.0;
        for (rbmd::Id it = 0; it < m; it++) {
          ddf1 = p * c - df1 * s;
          df1 = p * s + df1 * c;
          p = ddf1;
        }

        p = p * cos_shift + df1 * sin_shift;
        df1 = df1 * cos_shift - ddf1 * sin_shift;
        df1 *= -m;
        p += 1.0;
        if (m == 0) {
          p = 1.0 + cos_shift;
          df1 = 0.0;
        }

        const rbmd::Id min_gid = MIN(MIN(gid_i, gid_j), MIN(gid_k, gid_w));
        if (self_gid == min_gid) {
          local_energy_dihedral += k_coeff * p;
          local_count_dihedral += 1ULL;
        }

        const rbmd::Real fg = x12 * x23m + y12 * y23m + z12 * z23m;
        const rbmd::Real hg = x34 * x23m + y34 * y23m + z34 * z23m;
        const rbmd::Real fga = fg * ra2inv * rginv;
        const rbmd::Real hgb = hg * rb2inv * rginv;
        const rbmd::Real gaa = -ra2inv * rg;
        const rbmd::Real gbb = rb2inv * rg;

        const rbmd::Real dtfx = gaa * ax;
        const rbmd::Real dtfy = gaa * ay;
        const rbmd::Real dtfz = gaa * az;
        const rbmd::Real dtgx = fga * ax - hgb * bx;
        const rbmd::Real dtgy = fga * ay - hgb * by;
        const rbmd::Real dtgz = fga * az - hgb * bz;
        const rbmd::Real dthx = gbb * bx;
        const rbmd::Real dthy = gbb * by;
        const rbmd::Real dthz = gbb * bz;

        const rbmd::Real df = -k_coeff * df1;
        const rbmd::Real sx2 = df * dtgx;
        const rbmd::Real sy2 = df * dtgy;
        const rbmd::Real sz2 = df * dtgz;

        const rbmd::Real force_i_x = df * dtfx;
        const rbmd::Real force_i_y = df * dtfy;
        const rbmd::Real force_i_z = df * dtfz;

        const rbmd::Real force_j_x = sx2 - force_i_x;
        const rbmd::Real force_j_y = sy2 - force_i_y;
        const rbmd::Real force_j_z = sz2 - force_i_z;

        const rbmd::Real force_w_x = df * dthx;
        const rbmd::Real force_w_y = df * dthy;
        const rbmd::Real force_w_z = df * dthz;

        const rbmd::Real force_k_x = -sx2 - force_w_x;
        const rbmd::Real force_k_y = -sy2 - force_w_y;
        const rbmd::Real force_k_z = -sz2 - force_w_z;

        rbmd::Real global_virial_temp[6];
        global_virial_temp[0] = (x12 * force_i_x + x23 * force_k_x +
                                 (x34 + x23) * force_w_x);
        global_virial_temp[1] = (y12 * force_i_y + y23 * force_k_y +
                                 (y34 + y23) * force_w_y);
        global_virial_temp[2] = (z12 * force_i_z + z23 * force_k_z +
                                 (z34 + z23) * force_w_z);
        global_virial_temp[3] = (x12 * force_i_y + x23 * force_k_y +
                                 (x34 + x23) * force_w_y);
        global_virial_temp[4] = (x12 * force_i_z + x23 * force_k_z +
                                 (x34 + x23) * force_w_z);
        global_virial_temp[5] = (y12 * force_i_z + y23 * force_k_z +
                                 (y34 + y23) * force_w_z);

        rbmd::Real local_virial[6];
        local_virial[0] = 0.25 * global_virial_temp[0];
        local_virial[1] = 0.25 * global_virial_temp[1];
        local_virial[2] = 0.25 * global_virial_temp[2];
        local_virial[3] = 0.25 * global_virial_temp[3];
        local_virial[4] = 0.25 * global_virial_temp[4];
        local_virial[5] = 0.25 * global_virial_temp[5];

        if (self_gid == gid_i) {
          sum_fx += force_i_x;
          sum_fy += force_i_y;
          sum_fz += force_i_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_j) {
          sum_fx += force_j_x;
          sum_fy += force_j_y;
          sum_fz += force_j_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_k) {
          sum_fx += force_k_x;
          sum_fy += force_k_y;
          sum_fz += force_k_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_w) {
          sum_fx += force_w_x;
          sum_fy += force_w_y;
          sum_fz += force_w_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        }
      }

      fx[self_idx] = sum_fx;
      fy[self_idx] = sum_fy;
      fz[self_idx] = sum_fz;
      flat_virial[0 * num_local_atoms + self_idx] = sum_v0;
      flat_virial[1 * num_local_atoms + self_idx] = sum_v1;
      flat_virial[2 * num_local_atoms + self_idx] = sum_v2;
      flat_virial[3 * num_local_atoms + self_idx] = sum_v3;
      flat_virial[4 * num_local_atoms + self_idx] = sum_v4;
      flat_virial[5 * num_local_atoms + self_idx] = sum_v5;
    }

    const rbmd::Real block_sum_energy =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_energy)
            .Sum(local_energy_dihedral);
    const unsigned long long block_sum_count =
        BLOCKREDUCE<unsigned long long, BLOCK_SIZE>(temp_storage_count)
            .Sum(local_count_dihedral);

    if (threadIdx.x == 0) {
      atomicAdd(energy_dihedral, block_sum_energy);
      atomicAdd(count_dihedral, block_sum_count);
    }
  }

  __global__ void ComputeDihedralOPLSForcePerAtom(
      Box box, const rbmd::Id num_local_atoms,
      const rbmd::Id num_total_atoms,
      const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
      const rbmd::Real* dihedral_coeffs_k1,
      const rbmd::Real* dihedral_coeffs_k2,
      const rbmd::Real* dihedral_coeffs_k3,
      const rbmd::Real* dihedral_coeffs_k4, const int* d_num_dihedral,
      const int* d_dihedral_type, const rbmd::Id* d_dihedral_atom1,
      const rbmd::Id* d_dihedral_atom2, const rbmd::Id* d_dihedral_atom3,
      const rbmd::Id* d_dihedral_atom4, int dihedral_per_atom,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* flat_virial,
      rbmd::Real* energy_dihedral, unsigned long long* count_dihedral) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_energy;
    __shared__ typename BLOCKREDUCE<unsigned long long, BLOCK_SIZE>::TempStorage
        temp_storage_count;

    rbmd::Real local_energy_dihedral = 0.0;
    unsigned long long local_count_dihedral = 0ULL;

    const unsigned int self_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (self_idx < static_cast<unsigned int>(num_local_atoms)) {
      const rbmd::Id self_gid = atoms_id[self_idx];
      rbmd::Real sum_fx = 0.0;
      rbmd::Real sum_fy = 0.0;
      rbmd::Real sum_fz = 0.0;
      rbmd::Real sum_v0 = 0.0;
      rbmd::Real sum_v1 = 0.0;
      rbmd::Real sum_v2 = 0.0;
      rbmd::Real sum_v3 = 0.0;
      rbmd::Real sum_v4 = 0.0;
      rbmd::Real sum_v5 = 0.0;

      int num_dihedral_self = d_num_dihedral[self_idx];
      if (num_dihedral_self < 0) {
        num_dihedral_self = 0;
      }
      if (num_dihedral_self > dihedral_per_atom) {
        num_dihedral_self = dihedral_per_atom;
      }

      for (int k = 0; k < num_dihedral_self; ++k) {
        const int topo_offset =
            static_cast<int>(self_idx) * dihedral_per_atom + k;
        const int dihedraltype = d_dihedral_type[topo_offset];
        if (dihedraltype < 0) {
          continue;
        }

        const rbmd::Id gid_i = d_dihedral_atom1[topo_offset];
        const rbmd::Id gid_j = d_dihedral_atom2[topo_offset];
        const rbmd::Id gid_k = d_dihedral_atom3[topo_offset];
        const rbmd::Id gid_w = d_dihedral_atom4[topo_offset];

        const rbmd::Id idx_i = atom_id_to_idx[gid_i];
        const rbmd::Id idx_j = atom_id_to_idx[gid_j];
        const rbmd::Id idx_k = atom_id_to_idx[gid_k];
        const rbmd::Id idx_w = atom_id_to_idx[gid_w];
        if (idx_i < 0 || idx_j < 0 || idx_k < 0 || idx_w < 0 ||
            idx_i >= num_total_atoms || idx_j >= num_total_atoms ||
            idx_k >= num_total_atoms || idx_w >= num_total_atoms) {
          continue;
        }

        const rbmd::Real k1 = dihedral_coeffs_k1[dihedraltype];
        const rbmd::Real k2 = dihedral_coeffs_k2[dihedraltype];
        const rbmd::Real k3 = dihedral_coeffs_k3[dihedraltype];
        const rbmd::Real k4 = dihedral_coeffs_k4[dihedraltype];

        rbmd::Real x12 = px[idx_i] - px[idx_j];
        rbmd::Real y12 = py[idx_i] - py[idx_j];
        rbmd::Real z12 = pz[idx_i] - pz[idx_j];
        MinImageDistance_while(box, x12, y12, z12);

        rbmd::Real x23 = px[idx_k] - px[idx_j];
        rbmd::Real y23 = py[idx_k] - py[idx_j];
        rbmd::Real z23 = pz[idx_k] - pz[idx_j];
        MinImageDistance_while(box, x23, y23, z23);

        rbmd::Real x23m = -x23;
        rbmd::Real y23m = -y23;
        rbmd::Real z23m = -z23;

        rbmd::Real x34 = px[idx_w] - px[idx_k];
        rbmd::Real y34 = py[idx_w] - py[idx_k];
        rbmd::Real z34 = pz[idx_w] - pz[idx_k];
        MinImageDistance_while(box, x34, y34, z34);

        rbmd::Real sb1, sb2, sb3, rb1, rb3, c0, b1mag2, b1mag, b2mag2;
        rbmd::Real b2mag, b3mag2, b3mag, ctmp, r12c1, c1mag, r12c2;
        rbmd::Real c2mag, sc1, sc2, s1, s12, c, p, pd, a, a11, a22;
        rbmd::Real a33, a12, a13, a23, sx2, sy2, sz2;
        rbmd::Real s2, cx, cy, cz, cmag, dx, phi, si, siinv, sin2;

        sb1 = 1.0 / (x12 * x12 + y12 * y12 + z12 * z12);
        sb2 = 1.0 / (x23 * x23 + y23 * y23 + z23 * z23);
        sb3 = 1.0 / (x34 * x34 + y34 * y34 + z34 * z34);

        rb1 = SQRT(sb1);
        rb3 = SQRT(sb3);

        c0 = (x12 * x34 + y12 * y34 + z12 * z34) * rb1 * rb3;

        b1mag2 = x12 * x12 + y12 * y12 + z12 * z12;
        b1mag = SQRT(b1mag2);
        b2mag2 = x23 * x23 + y23 * y23 + z23 * z23;
        b2mag = SQRT(b2mag2);
        b3mag2 = x34 * x34 + y34 * y34 + z34 * z34;
        b3mag = SQRT(b3mag2);

        ctmp = x12 * x23 + y12 * y23 + z12 * z23;
        r12c1 = 1.0 / (b1mag * b2mag);
        c1mag = ctmp * r12c1;

        ctmp = x23m * x34 + y23m * y34 + z23m * z34;
        r12c2 = 1.0 / (b2mag * b3mag);
        c2mag = ctmp * r12c2;

        sin2 = MAX(1.0 - c1mag * c1mag, 0.0);
        sc1 = SQRT(sin2);
        if (sc1 < SMALL) sc1 = SMALL;
        sc1 = 1.0 / sc1;

        sin2 = MAX(1.0 - c2mag * c2mag, 0.0);
        sc2 = SQRT(sin2);
        if (sc2 < SMALL) sc2 = SMALL;
        sc2 = 1.0 / sc2;

        s1 = sc1 * sc1;
        s2 = sc2 * sc2;
        s12 = sc1 * sc2;
        c = (c0 + c1mag * c2mag) * s12;

        cx = y12 * z23 - z12 * y23;
        cy = z12 * x23 - x12 * z23;
        cz = x12 * y23 - y12 * x23;
        cmag = SQRT(cx * cx + cy * cy + cz * cz);
        dx = (cx * x34 + cy * y34 + cz * z34) / cmag / b3mag;

        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        phi = ACOS(c);
        if (dx < 0.0) phi *= -1.0;
        si = SIN(phi);
        if (fabs(si) < SMALLER) si = SMALLER;
        siinv = 1.0 / si;

        p = k1 * (1.0 + c) + k2 * (1.0 - COS(2.0 * phi)) +
            k3 * (1.0 + COS(3.0 * phi)) + k4 * (1.0 - COS(4.0 * phi));
        pd = k1 - 2.0 * k2 * SIN(2.0 * phi) * siinv +
             3.0 * k3 * SIN(3.0 * phi) * siinv -
             4.0 * k4 * SIN(4.0 * phi) * siinv;

        const rbmd::Id min_gid = MIN(MIN(gid_i, gid_j), MIN(gid_k, gid_w));
        if (self_gid == min_gid) {
          local_energy_dihedral += p;
          local_count_dihedral += 1ULL;
        }

        a = pd;
        c = c * a;
        s12 = s12 * a;
        a11 = c * sb1 * s1;
        a22 = -sb2 * (2.0 * c0 * s12 - c * (s1 + s2));
        a33 = c * sb3 * s2;
        a12 = -r12c1 * (c1mag * c * s1 + c2mag * s12);
        a13 = -rb1 * rb3 * s12;
        a23 = r12c2 * (c2mag * c * s2 + c1mag * s12);

        sx2 = a12 * x12 + a22 * x23 + a23 * x34;
        sy2 = a12 * y12 + a22 * y23 + a23 * y34;
        sz2 = a12 * z12 + a22 * z23 + a23 * z34;

        const rbmd::Real force_i_x = a11 * x12 + a12 * x23 + a13 * x34;
        const rbmd::Real force_i_y = a11 * y12 + a12 * y23 + a13 * y34;
        const rbmd::Real force_i_z = a11 * z12 + a12 * z23 + a13 * z34;

        const rbmd::Real force_j_x = -sx2 - force_i_x;
        const rbmd::Real force_j_y = -sy2 - force_i_y;
        const rbmd::Real force_j_z = -sz2 - force_i_z;

        const rbmd::Real force_w_x = a13 * x12 + a23 * x23 + a33 * x34;
        const rbmd::Real force_w_y = a13 * y12 + a23 * y23 + a33 * y34;
        const rbmd::Real force_w_z = a13 * z12 + a23 * z23 + a33 * z34;

        const rbmd::Real force_k_x = sx2 - force_w_x;
        const rbmd::Real force_k_y = sy2 - force_w_y;
        const rbmd::Real force_k_z = sz2 - force_w_z;

        rbmd::Real global_virial_temp[6];
        global_virial_temp[0] = (x12 * force_i_x + x23 * force_k_x +
                                 (x34 + x23) * force_w_x);
        global_virial_temp[1] = (y12 * force_i_y + y23 * force_k_y +
                                 (y34 + y23) * force_w_y);
        global_virial_temp[2] = (z12 * force_i_z + z23 * force_k_z +
                                 (z34 + z23) * force_w_z);
        global_virial_temp[3] = (x12 * force_i_y + x23 * force_k_y +
                                 (x34 + x23) * force_w_y);
        global_virial_temp[4] = (x12 * force_i_z + x23 * force_k_z +
                                 (x34 + x23) * force_w_z);
        global_virial_temp[5] = (y12 * force_i_z + y23 * force_k_z +
                                 (y34 + y23) * force_w_z);

        rbmd::Real local_virial[6];
        local_virial[0] = 0.25 * global_virial_temp[0];
        local_virial[1] = 0.25 * global_virial_temp[1];
        local_virial[2] = 0.25 * global_virial_temp[2];
        local_virial[3] = 0.25 * global_virial_temp[3];
        local_virial[4] = 0.25 * global_virial_temp[4];
        local_virial[5] = 0.25 * global_virial_temp[5];

        if (self_gid == gid_i) {
          sum_fx += force_i_x;
          sum_fy += force_i_y;
          sum_fz += force_i_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_j) {
          sum_fx += force_j_x;
          sum_fy += force_j_y;
          sum_fz += force_j_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_k) {
          sum_fx += force_k_x;
          sum_fy += force_k_y;
          sum_fz += force_k_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_w) {
          sum_fx += force_w_x;
          sum_fy += force_w_y;
          sum_fz += force_w_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        }
      }

      fx[self_idx] = sum_fx;
      fy[self_idx] = sum_fy;
      fz[self_idx] = sum_fz;
      flat_virial[0 * num_local_atoms + self_idx] = sum_v0;
      flat_virial[1 * num_local_atoms + self_idx] = sum_v1;
      flat_virial[2 * num_local_atoms + self_idx] = sum_v2;
      flat_virial[3 * num_local_atoms + self_idx] = sum_v3;
      flat_virial[4 * num_local_atoms + self_idx] = sum_v4;
      flat_virial[5 * num_local_atoms + self_idx] = sum_v5;
    }

    const rbmd::Real block_sum_energy =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_energy)
            .Sum(local_energy_dihedral);
    const unsigned long long block_sum_count =
        BLOCKREDUCE<unsigned long long, BLOCK_SIZE>(temp_storage_count)
            .Sum(local_count_dihedral);

    if (threadIdx.x == 0) {
      atomicAdd(energy_dihedral, block_sum_energy);
      atomicAdd(count_dihedral, block_sum_count);
    }
  }

  __global__ void ComputeImproperHarmonicForcePerAtom(
      Box box, const rbmd::Id num_local_atoms,
      const rbmd::Id num_total_atoms,
      const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
      const rbmd::Real* improper_coeffs_k, const rbmd::Real* improper_coeffs_chi,
      const int* d_num_improper, const int* d_improper_type,
      const rbmd::Id* d_improper_atom1, const rbmd::Id* d_improper_atom2,
      const rbmd::Id* d_improper_atom3, const rbmd::Id* d_improper_atom4,
      int improper_per_atom,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* flat_virial,
      rbmd::Real* energy_improper, unsigned long long* count_improper) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_energy;
    __shared__ typename BLOCKREDUCE<unsigned long long, BLOCK_SIZE>::TempStorage
        temp_storage_count;

    rbmd::Real local_energy_improper = 0.0;
    unsigned long long local_count_improper = 0ULL;

    const unsigned int self_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (self_idx < static_cast<unsigned int>(num_local_atoms)) {
      const rbmd::Id self_gid = atoms_id[self_idx];
      rbmd::Real sum_fx = 0.0;
      rbmd::Real sum_fy = 0.0;
      rbmd::Real sum_fz = 0.0;
      rbmd::Real sum_v0 = 0.0;
      rbmd::Real sum_v1 = 0.0;
      rbmd::Real sum_v2 = 0.0;
      rbmd::Real sum_v3 = 0.0;
      rbmd::Real sum_v4 = 0.0;
      rbmd::Real sum_v5 = 0.0;

      int num_improper_self = d_num_improper[self_idx];
      if (num_improper_self < 0) {
        num_improper_self = 0;
      }
      if (num_improper_self > improper_per_atom) {
        num_improper_self = improper_per_atom;
      }

      for (int k = 0; k < num_improper_self; ++k) {
        const int topo_offset =
            static_cast<int>(self_idx) * improper_per_atom + k;
        const int impropertype = d_improper_type[topo_offset];
        if (impropertype < 0) {
          continue;
        }

        const rbmd::Id gid_i = d_improper_atom1[topo_offset];
        const rbmd::Id gid_j = d_improper_atom2[topo_offset];
        const rbmd::Id gid_k = d_improper_atom3[topo_offset];
        const rbmd::Id gid_w = d_improper_atom4[topo_offset];

        const rbmd::Id idx_i = atom_id_to_idx[gid_i];
        const rbmd::Id idx_j = atom_id_to_idx[gid_j];
        const rbmd::Id idx_k = atom_id_to_idx[gid_k];
        const rbmd::Id idx_w = atom_id_to_idx[gid_w];
        if (idx_i < 0 || idx_j < 0 || idx_k < 0 || idx_w < 0 ||
            idx_i >= num_total_atoms || idx_j >= num_total_atoms ||
            idx_k >= num_total_atoms || idx_w >= num_total_atoms) {
          continue;
        }

        const rbmd::Real coeff_k = improper_coeffs_k[impropertype];
        const rbmd::Real chi = improper_coeffs_chi[impropertype];

        rbmd::Real x12 = px[idx_i] - px[idx_j];
        rbmd::Real y12 = py[idx_i] - py[idx_j];
        rbmd::Real z12 = pz[idx_i] - pz[idx_j];
        MinImageDistance_while(box, x12, y12, z12);

        rbmd::Real x23 = px[idx_k] - px[idx_j];
        rbmd::Real y23 = py[idx_k] - py[idx_j];
        rbmd::Real z23 = pz[idx_k] - pz[idx_j];
        MinImageDistance_while(box, x23, y23, z23);

        rbmd::Real x34 = px[idx_w] - px[idx_k];
        rbmd::Real y34 = py[idx_w] - py[idx_k];
        rbmd::Real z34 = pz[idx_w] - pz[idx_k];
        MinImageDistance_while(box, x34, y34, z34);

        const rbmd::Real ss1 = 1.0 / (x12 * x12 + y12 * y12 + z12 * z12);
        const rbmd::Real ss2 = 1.0 / (x23 * x23 + y23 * y23 + z23 * z23);
        const rbmd::Real ss3 = 1.0 / (x34 * x34 + y34 * y34 + z34 * z34);

        const rbmd::Real r1 = SQRT(ss1);
        const rbmd::Real r2 = SQRT(ss2);
        const rbmd::Real r3 = SQRT(ss3);

        const rbmd::Real c0 = (x12 * x34 + y12 * y34 + z12 * z34) * r1 * r3;
        const rbmd::Real c1 = (x12 * x23 + y12 * y23 + z12 * z23) * r1 * r2;
        const rbmd::Real c2 = -(x34 * x23 + y34 * y23 + z34 * z23) * r3 * r2;

        rbmd::Real s1 = 1.0 - c1 * c1;
        if (s1 < SMALL) s1 = SMALL;
        s1 = 1.0 / s1;

        rbmd::Real s2 = 1.0 - c2 * c2;
        if (s2 < SMALL) s2 = SMALL;
        s2 = 1.0 / s2;

        rbmd::Real s12 = SQRT(s1 * s2);
        rbmd::Real c = (c1 * c2 + c0) * s12;
        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        rbmd::Real s = SQRT(1.0 - c * c);
        if (s < SMALL) s = SMALL;

        const rbmd::Real domega = ACOS(c) - chi;
        rbmd::Real a = coeff_k * domega;
        const rbmd::Real improper_energy = a * domega;

        const rbmd::Id min_gid = MIN(MIN(gid_i, gid_j), MIN(gid_k, gid_w));
        if (self_gid == min_gid) {
          local_energy_improper += improper_energy;
          local_count_improper += 1ULL;
        }

        a = -a * 2.0 / s;
        c = c * a;
        s12 = s12 * a;
        const rbmd::Real a11 = c * ss1 * s1;
        const rbmd::Real a22 = -ss2 * (2.0 * c0 * s12 - c * (s1 + s2));
        const rbmd::Real a33 = c * ss3 * s2;
        const rbmd::Real a12 = -r1 * r2 * (c1 * c * s1 + c2 * s12);
        const rbmd::Real a13 = -r1 * r3 * s12;
        const rbmd::Real a23 = r2 * r3 * (c2 * c * s2 + c1 * s12);

        const rbmd::Real sx2 = a22 * x23 + a23 * x34 + a12 * x12;
        const rbmd::Real sy2 = a22 * y23 + a23 * y34 + a12 * y12;
        const rbmd::Real sz2 = a22 * z23 + a23 * z34 + a12 * z12;

        const rbmd::Real force_i_x = a12 * x23 + a13 * x34 + a11 * x12;
        const rbmd::Real force_i_y = a12 * y23 + a13 * y34 + a11 * y12;
        const rbmd::Real force_i_z = a12 * z23 + a13 * z34 + a11 * z12;

        const rbmd::Real force_j_x = -sx2 - force_i_x;
        const rbmd::Real force_j_y = -sy2 - force_i_y;
        const rbmd::Real force_j_z = -sz2 - force_i_z;

        const rbmd::Real force_w_x = a23 * x23 + a33 * x34 + a13 * x12;
        const rbmd::Real force_w_y = a23 * y23 + a33 * y34 + a13 * y12;
        const rbmd::Real force_w_z = a23 * z23 + a33 * z34 + a13 * z12;

        const rbmd::Real force_k_x = sx2 - force_w_x;
        const rbmd::Real force_k_y = sy2 - force_w_y;
        const rbmd::Real force_k_z = sz2 - force_w_z;

        rbmd::Real global_virial_temp[6];
        global_virial_temp[0] = (x12 * force_i_x + x23 * force_k_x +
                                 (x34 + x23) * force_w_x);
        global_virial_temp[1] = (y12 * force_i_y + y23 * force_k_y +
                                 (y34 + y23) * force_w_y);
        global_virial_temp[2] = (z12 * force_i_z + z23 * force_k_z +
                                 (z34 + z23) * force_w_z);
        global_virial_temp[3] = (x12 * force_i_y + x23 * force_k_y +
                                 (x34 + x23) * force_w_y);
        global_virial_temp[4] = (x12 * force_i_z + x23 * force_k_z +
                                 (x34 + x23) * force_w_z);
        global_virial_temp[5] = (y12 * force_i_z + y23 * force_k_z +
                                 (y34 + y23) * force_w_z);

        rbmd::Real local_virial[6];
        local_virial[0] = 0.25 * global_virial_temp[0];
        local_virial[1] = 0.25 * global_virial_temp[1];
        local_virial[2] = 0.25 * global_virial_temp[2];
        local_virial[3] = 0.25 * global_virial_temp[3];
        local_virial[4] = 0.25 * global_virial_temp[4];
        local_virial[5] = 0.25 * global_virial_temp[5];

        if (self_gid == gid_i) {
          sum_fx += force_i_x;
          sum_fy += force_i_y;
          sum_fz += force_i_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_j) {
          sum_fx += force_j_x;
          sum_fy += force_j_y;
          sum_fz += force_j_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_k) {
          sum_fx += force_k_x;
          sum_fy += force_k_y;
          sum_fz += force_k_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_w) {
          sum_fx += force_w_x;
          sum_fy += force_w_y;
          sum_fz += force_w_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        }
      }

      fx[self_idx] = sum_fx;
      fy[self_idx] = sum_fy;
      fz[self_idx] = sum_fz;
      flat_virial[0 * num_local_atoms + self_idx] = sum_v0;
      flat_virial[1 * num_local_atoms + self_idx] = sum_v1;
      flat_virial[2 * num_local_atoms + self_idx] = sum_v2;
      flat_virial[3 * num_local_atoms + self_idx] = sum_v3;
      flat_virial[4 * num_local_atoms + self_idx] = sum_v4;
      flat_virial[5 * num_local_atoms + self_idx] = sum_v5;
    }

    const rbmd::Real block_sum_energy =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_energy)
            .Sum(local_energy_improper);
    const unsigned long long block_sum_count =
        BLOCKREDUCE<unsigned long long, BLOCK_SIZE>(temp_storage_count)
            .Sum(local_count_improper);

    if (threadIdx.x == 0) {
      atomicAdd(energy_improper, block_sum_energy);
      atomicAdd(count_improper, block_sum_count);
    }
  }

  __global__ void ComputeImproperCVFFForcePerAtom(
      Box box, const rbmd::Id num_local_atoms,
      const rbmd::Id num_total_atoms,
      const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
      const rbmd::Real* improper_coeffs_k, const rbmd::Id* improper_coeffs_d,
      const rbmd::Id* improper_coeffs_n, const int* d_num_improper,
      const int* d_improper_type, const rbmd::Id* d_improper_atom1,
      const rbmd::Id* d_improper_atom2, const rbmd::Id* d_improper_atom3,
      const rbmd::Id* d_improper_atom4, int improper_per_atom,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* flat_virial,
      rbmd::Real* energy_improper, unsigned long long* count_improper) {
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_energy;
    __shared__ typename BLOCKREDUCE<unsigned long long, BLOCK_SIZE>::TempStorage
        temp_storage_count;

    rbmd::Real local_energy_improper = 0.0;
    unsigned long long local_count_improper = 0ULL;

    const unsigned int self_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (self_idx < static_cast<unsigned int>(num_local_atoms)) {
      const rbmd::Id self_gid = atoms_id[self_idx];
      rbmd::Real sum_fx = 0.0;
      rbmd::Real sum_fy = 0.0;
      rbmd::Real sum_fz = 0.0;
      rbmd::Real sum_v0 = 0.0;
      rbmd::Real sum_v1 = 0.0;
      rbmd::Real sum_v2 = 0.0;
      rbmd::Real sum_v3 = 0.0;
      rbmd::Real sum_v4 = 0.0;
      rbmd::Real sum_v5 = 0.0;

      int num_improper_self = d_num_improper[self_idx];
      if (num_improper_self < 0) {
        num_improper_self = 0;
      }
      if (num_improper_self > improper_per_atom) {
        num_improper_self = improper_per_atom;
      }

      for (int k = 0; k < num_improper_self; ++k) {
        const int topo_offset =
            static_cast<int>(self_idx) * improper_per_atom + k;
        const int impropertype = d_improper_type[topo_offset];
        if (impropertype < 0) {
          continue;
        }

        const rbmd::Id gid_i = d_improper_atom1[topo_offset];
        const rbmd::Id gid_j = d_improper_atom2[topo_offset];
        const rbmd::Id gid_k = d_improper_atom3[topo_offset];
        const rbmd::Id gid_w = d_improper_atom4[topo_offset];

        const rbmd::Id idx_i = atom_id_to_idx[gid_i];
        const rbmd::Id idx_j = atom_id_to_idx[gid_j];
        const rbmd::Id idx_k = atom_id_to_idx[gid_k];
        const rbmd::Id idx_w = atom_id_to_idx[gid_w];
        if (idx_i < 0 || idx_j < 0 || idx_k < 0 || idx_w < 0 ||
            idx_i >= num_total_atoms || idx_j >= num_total_atoms ||
            idx_k >= num_total_atoms || idx_w >= num_total_atoms) {
          continue;
        }

        const rbmd::Real coeff_k = improper_coeffs_k[impropertype];
        const rbmd::Id d = improper_coeffs_d[impropertype];
        const rbmd::Id n = improper_coeffs_n[impropertype];

        rbmd::Real x12 = px[idx_i] - px[idx_j];
        rbmd::Real y12 = py[idx_i] - py[idx_j];
        rbmd::Real z12 = pz[idx_i] - pz[idx_j];
        MinImageDistance_while(box, x12, y12, z12);

        rbmd::Real x23 = px[idx_k] - px[idx_j];
        rbmd::Real y23 = py[idx_k] - py[idx_j];
        rbmd::Real z23 = pz[idx_k] - pz[idx_j];
        MinImageDistance_while(box, x23, y23, z23);

        rbmd::Real x23m = -x23;
        rbmd::Real y23m = -y23;
        rbmd::Real z23m = -z23;

        rbmd::Real x34 = px[idx_w] - px[idx_k];
        rbmd::Real y34 = py[idx_w] - py[idx_k];
        rbmd::Real z34 = pz[idx_w] - pz[idx_k];
        MinImageDistance_while(box, x34, y34, z34);

        const rbmd::Real sb1 = 1.0 / (x12 * x12 + y12 * y12 + z12 * z12);
        const rbmd::Real sb2 = 1.0 / (x23 * x23 + y23 * y23 + z23 * z23);
        const rbmd::Real sb3 = 1.0 / (x34 * x34 + y34 * y34 + z34 * z34);

        const rbmd::Real rb1 = SQRT(sb1);
        const rbmd::Real rb3 = SQRT(sb3);
        const rbmd::Real c0 = (x12 * x34 + y12 * y34 + z12 * z34) * rb1 * rb3;

        rbmd::Real b1mag2 = x12 * x12 + y12 * y12 + z12 * z12;
        rbmd::Real b1mag = SQRT(b1mag2);
        rbmd::Real b2mag2 = x23 * x23 + y23 * y23 + z23 * z23;
        rbmd::Real b2mag = SQRT(b2mag2);
        rbmd::Real b3mag2 = x34 * x34 + y34 * y34 + z34 * z34;
        rbmd::Real b3mag = SQRT(b3mag2);

        rbmd::Real ctmp = x12 * x23 + y12 * y23 + z12 * z23;
        rbmd::Real r12c1 = 1.0 / (b1mag * b2mag);
        rbmd::Real c1mag = ctmp * r12c1;

        ctmp = x23m * x34 + y23m * y34 + z23m * z34;
        rbmd::Real r12c2 = 1.0 / (b2mag * b3mag);
        rbmd::Real c2mag = ctmp * r12c2;

        rbmd::Real sc1 = SQRT(1.0 - c1mag * c1mag);
        if (sc1 < SMALL) sc1 = SMALL;
        sc1 = 1.0 / sc1;

        rbmd::Real sc2 = SQRT(1.0 - c2mag * c2mag);
        if (sc2 < SMALL) sc2 = SMALL;
        sc2 = 1.0 / sc2;

        rbmd::Real s1 = sc1 * sc1;
        rbmd::Real s2 = sc2 * sc2;
        rbmd::Real s12 = sc1 * sc2;
        rbmd::Real c = (c0 + c1mag * c2mag) * s12;
        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        rbmd::Real p = 2.0;
        rbmd::Real pd = 0.0;
        rbmd::Real rc2 = 0.0;
        const rbmd::Id m = n;
        if (m == 2) {
          p = 2.0 * c * c;
          pd = 2.0 * c;
        } else if (m == 3) {
          rc2 = c * c;
          p = (4.0 * rc2 - 3.0) * c + 1.0;
          pd = 6.0 * rc2 - 1.5;
        } else if (m == 4) {
          rc2 = c * c;
          p = 8.0 * (rc2 - 1) * rc2 + 2.0;
          pd = (16.0 * rc2 - 8.0) * c;
        } else if (m == 6) {
          rc2 = c * c;
          p = ((32.0 * rc2 - 48.0) * rc2 + 18.0) * rc2;
          pd = (96.0 * (rc2 - 1.0) * rc2 + 18.0) * c;
        } else if (m == 1) {
          p = c + 1.0;
          pd = 0.5;
        } else if (m == 5) {
          rc2 = c * c;
          p = ((16.0 * rc2 - 20.0) * rc2 + 5.0) * c + 1.0;
          pd = (40.0 * rc2 - 30.0) * rc2 + 2.5;
        } else if (m == 0) {
          p = 2.0;
          pd = 0.0;
        }

        if (d == -1) {
          p = 2.0 - p;
          pd = -pd;
        }

        const rbmd::Id min_gid = MIN(MIN(gid_i, gid_j), MIN(gid_k, gid_w));
        if (self_gid == min_gid) {
          local_energy_improper += coeff_k * p;
          local_count_improper += 1ULL;
        }

        rbmd::Real a = 2.0 * coeff_k * pd;
        c = c * a;
        s12 = s12 * a;
        const rbmd::Real a11 = c * sb1 * s1;
        const rbmd::Real a22 = -sb2 * (2.0 * c0 * s12 - c * (s1 + s2));
        const rbmd::Real a33 = c * sb3 * s2;
        const rbmd::Real a12 = -r12c1 * (c1mag * c * s1 + c2mag * s12);
        const rbmd::Real a13 = -rb1 * rb3 * s12;
        const rbmd::Real a23 = r12c2 * (c2mag * c * s2 + c1mag * s12);

        const rbmd::Real sx2 = a12 * x12 + a22 * x23 + a23 * x34;
        const rbmd::Real sy2 = a12 * y12 + a22 * y23 + a23 * y34;
        const rbmd::Real sz2 = a12 * z12 + a22 * z23 + a23 * z34;

        const rbmd::Real force_i_x = a11 * x12 + a12 * x23 + a13 * x34;
        const rbmd::Real force_i_y = a11 * y12 + a12 * y23 + a13 * y34;
        const rbmd::Real force_i_z = a11 * z12 + a12 * z23 + a13 * z34;

        const rbmd::Real force_j_x = -sx2 - force_i_x;
        const rbmd::Real force_j_y = -sy2 - force_i_y;
        const rbmd::Real force_j_z = -sz2 - force_i_z;

        const rbmd::Real force_w_x = a13 * x12 + a23 * x23 + a33 * x34;
        const rbmd::Real force_w_y = a13 * y12 + a23 * y23 + a33 * y34;
        const rbmd::Real force_w_z = a13 * z12 + a23 * z23 + a33 * z34;

        const rbmd::Real force_k_x = sx2 - force_w_x;
        const rbmd::Real force_k_y = sy2 - force_w_y;
        const rbmd::Real force_k_z = sz2 - force_w_z;

        rbmd::Real global_virial_temp[6];
        global_virial_temp[0] = (x12 * force_i_x + x23 * force_k_x +
                                 (x34 + x23) * force_w_x);
        global_virial_temp[1] = (y12 * force_i_y + y23 * force_k_y +
                                 (y34 + y23) * force_w_y);
        global_virial_temp[2] = (z12 * force_i_z + z23 * force_k_z +
                                 (z34 + z23) * force_w_z);
        global_virial_temp[3] = (x12 * force_i_y + x23 * force_k_y +
                                 (x34 + x23) * force_w_y);
        global_virial_temp[4] = (x12 * force_i_z + x23 * force_k_z +
                                 (x34 + x23) * force_w_z);
        global_virial_temp[5] = (y12 * force_i_z + y23 * force_k_z +
                                 (y34 + y23) * force_w_z);

        rbmd::Real local_virial[6];
        local_virial[0] = 0.25 * global_virial_temp[0];
        local_virial[1] = 0.25 * global_virial_temp[1];
        local_virial[2] = 0.25 * global_virial_temp[2];
        local_virial[3] = 0.25 * global_virial_temp[3];
        local_virial[4] = 0.25 * global_virial_temp[4];
        local_virial[5] = 0.25 * global_virial_temp[5];

        if (self_gid == gid_i) {
          sum_fx += force_i_x;
          sum_fy += force_i_y;
          sum_fz += force_i_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_j) {
          sum_fx += force_j_x;
          sum_fy += force_j_y;
          sum_fz += force_j_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_k) {
          sum_fx += force_k_x;
          sum_fy += force_k_y;
          sum_fz += force_k_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        } else if (self_gid == gid_w) {
          sum_fx += force_w_x;
          sum_fy += force_w_y;
          sum_fz += force_w_z;
          sum_v0 += local_virial[0];
          sum_v1 += local_virial[1];
          sum_v2 += local_virial[2];
          sum_v3 += local_virial[3];
          sum_v4 += local_virial[4];
          sum_v5 += local_virial[5];
        }
      }

      fx[self_idx] = sum_fx;
      fy[self_idx] = sum_fy;
      fz[self_idx] = sum_fz;
      flat_virial[0 * num_local_atoms + self_idx] = sum_v0;
      flat_virial[1 * num_local_atoms + self_idx] = sum_v1;
      flat_virial[2 * num_local_atoms + self_idx] = sum_v2;
      flat_virial[3 * num_local_atoms + self_idx] = sum_v3;
      flat_virial[4 * num_local_atoms + self_idx] = sum_v4;
      flat_virial[5 * num_local_atoms + self_idx] = sum_v5;
    }

    const rbmd::Real block_sum_energy =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_energy)
            .Sum(local_energy_improper);
    const unsigned long long block_sum_count =
        BLOCKREDUCE<unsigned long long, BLOCK_SIZE>(temp_storage_count)
            .Sum(local_count_improper);

    if (threadIdx.x == 0) {
      atomicAdd(energy_improper, block_sum_energy);
      atomicAdd(count_improper, block_sum_count);
    }
  }

////////////////////////////////////////
//verlet-list:  force  of special lJ_cut and coul_cut
  void SpecialLJCutCoulForceOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Real cut_off,
      const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
      const rbmd::Id lj_num_types, const rbmd::Real* lj_force12,
      const rbmd::Real* lj_force6, const rbmd::Id* start_id,
      const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
      const int* nspecial, const rbmd::Id* special, int maxspecial,
      rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
      const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
      const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* flat_virial,rbmd::Real* total_evdwl, rbmd::Real* total_ecoul) {
    unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(
        ComputeSpecialLJCutCoulForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
            box, cut_off, num_atoms, alpha, qqr2e, atoms_type,
            atoms_id, lj_num_types, lj_force12, lj_force6, start_id, end_id,
            id_verletlist, nspecial,
            special, maxspecial, w12, w13, w14, charge, px, py, pz, fx, fy,
            fz, flat_virial,total_evdwl, total_ecoul));
  }

//RBL:  force  of special lJ_cut and coul_cut
  void SpecialLJCutCoulRBLForceOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Real rs, const rbmd::Real rc,
      const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
      const rbmd::Id pice_num, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
      const rbmd::Id lj_num_types, const rbmd::Real* lj_force12,
      const rbmd::Real* lj_force6, const rbmd::Id* start_id,
      const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
      const rbmd::Id* id_random_neighbor, const rbmd::Id* random_neighbor_num,
      const int* nspecial, const rbmd::Id* special, int maxspecial,
      rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
      const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
      const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz) {
    unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(
        ComputeSpecialLJCutCoulRBLForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
            box, rs, rc, num_atoms, neighbor_sample_num, pice_num,
            alpha, qqr2e, atoms_type, atoms_id, lj_num_types, lj_force12,
            lj_force6, start_id, end_id, id_verletlist, id_random_neighbor,
            random_neighbor_num, nspecial,
            special, maxspecial, w12, w13, w14, charge, px, py, pz, fx, fy,
            fz));
  }

  //verlet-list:  Energy of special lJ_cut and coul_cut
  void SpeciaLJCutCoulEnergyOp<device::DEVICE_GPU>::operator()(
     Box box, const rbmd::Real cut_off,
    const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
    const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
    const rbmd::Real* sigma, const rbmd::Real* eps, const rbmd::Id* start_id,
    const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
    const int* nspecial, const rbmd::Id* special, int maxspecial,
    rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
    const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* flat_virial,
    rbmd::Real* total_evdwl, rbmd::Real* total_ecoul) {
    unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(
        ComputeSpecialLJCutCoulEnergy<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
            box, cut_off, num_atoms, alpha, qqr2e, atoms_type,
            atoms_id, sigma, eps, start_id, end_id, id_verletlist, nspecial,
            special, maxspecial, w12, w13, w14, charge, px, py, pz,
            flat_virial,total_evdwl, total_ecoul));
  }

  // force of bond
  void ComputeBondForceOp<device::DEVICE_GPU>::operator()(
       Box box,const rbmd::Id num_atoms,const rbmd::Id num_bonds,
       const rbmd::Id* atom_id_to_idx,const rbmd::Real* bond_coeffs_k,
       const rbmd::Real* bond_coeffs_equilibrium,const rbmd::Id* bond_type,
       const rbmd::Id* bondlisti,const rbmd::Id* bondlistj,
       const rbmd::Real* px, const rbmd::Real* py,const rbmd::Real* pz,
       rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
       rbmd::Real* flat_virial, rbmd::Real* global_virial,rbmd::Real* energy_bond) {
    unsigned int blocks_per_grid = (num_bonds + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeBondForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, num_atoms,num_bonds, atom_id_to_idx, bond_coeffs_k, bond_coeffs_equilibrium,
        bond_type, bondlisti, bondlistj, px, py, pz, fx, fy, fz,flat_virial,
        global_virial,energy_bond));
  }

  void ComputeBondForcePerAtomOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
       const rbmd::Id* atoms_id,
       const rbmd::Id* atom_id_to_idx, const rbmd::Id atom_id_to_idx_size,
       const rbmd::Real* bond_coeffs_k,
       int num_bond_types, const rbmd::Real* bond_coeffs_equilibrium,
       const int* d_num_bond, const int* d_bond_type, const rbmd::Id* d_bond_atom,
       int bond_per_atom, const rbmd::Real* px, const rbmd::Real* py,
       const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
       rbmd::Real* energy_bond, unsigned long long* count_bond) {
    unsigned int blocks_per_grid =
        (num_local_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeBondForcePerAtom<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, num_local_atoms, num_total_atoms, atoms_id, atom_id_to_idx,
        atom_id_to_idx_size,
        bond_coeffs_k,
        num_bond_types, bond_coeffs_equilibrium, d_num_bond, d_bond_type,
        d_bond_atom, bond_per_atom, px, py, pz, fx, fy, fz, energy_bond,
        count_bond));
  }

// force of angle
  void ComputeAngleForceOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Id num_atoms,const rbmd::Id num_anglels,
       const rbmd::Id* _atom_id_to_idx,
      const rbmd::Real* anglel_coeffs_k,
      const rbmd::Real* anglel_coeffs_equilibrium, const rbmd::Id* anglel_type,
      const rbmd::Id* anglelisti, const rbmd::Id* anglelistj,
      const rbmd::Id* anglelistk, const rbmd::Real* px, const rbmd::Real* py,
      const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* flat_virial,rbmd::Real* global_virial, rbmd::Real* energy_angle) {
    unsigned int blocks_per_grid = (num_anglels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeAngleForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, num_atoms,num_anglels, _atom_id_to_idx, anglel_coeffs_k,
        anglel_coeffs_equilibrium, anglel_type, anglelisti, anglelistj,
        anglelistk, px, py, pz, fx, fy, fz, flat_virial,global_virial,energy_angle));
  }

  void ComputeAngleForcePerAtomOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
       const rbmd::Id* atoms_id,
       const rbmd::Id* atom_id_to_idx, const rbmd::Real* anglel_coeffs_k,
       const rbmd::Real* anglel_coeffs_equilibrium, const int* d_num_angle,
       const int* d_angle_type, const rbmd::Id* d_angle_atom1,
       const rbmd::Id* d_angle_atom2, const rbmd::Id* d_angle_atom3,
       int angle_per_atom, const rbmd::Real* px, const rbmd::Real* py,
       const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
       rbmd::Real* energy_angle, unsigned long long* count_angle) {
    unsigned int blocks_per_grid =
        (num_local_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeAngleForcePerAtom<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, num_local_atoms, num_total_atoms, atoms_id, atom_id_to_idx,
        anglel_coeffs_k,
        anglel_coeffs_equilibrium, d_num_angle, d_angle_type, d_angle_atom1,
        d_angle_atom2, d_angle_atom3, angle_per_atom, px, py, pz, fx, fy, fz,
        energy_angle, count_angle));
  }


// force of dihedral
  void ComputeDihedralForceOp<device::DEVICE_GPU>::operator()(
       Box box,const rbmd::Id num_atoms,const rbmd::Id num_dihedrals,
       const rbmd::Id* atom_id_to_idx,const rbmd::Real* dihedral_coeffs_k,
       const rbmd::Id* dihedral_coeffs_sign,
       const rbmd::Id* dihedral_coeffs_multiplicity, const rbmd::Id* dihedral_type,
       const rbmd::Id* dihedrallisti, const rbmd::Id* dihedrallistj,
      const rbmd::Id* dihedrallistk, const rbmd::Id* dihedrallistw,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,rbmd::Real* flat_virial,
      rbmd::Real* global_virial,rbmd::Real* energy_dihedral) {
    unsigned int blocks_per_grid = (num_dihedrals + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeDihedralForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, num_atoms,num_dihedrals, atom_id_to_idx, dihedral_coeffs_k,
        dihedral_coeffs_sign, dihedral_coeffs_multiplicity, dihedral_type,
        dihedrallisti, dihedrallistj, dihedrallistk, dihedrallistw, px, py, pz,
        fx, fy, fz, flat_virial,global_virial,energy_dihedral));
  }

  void ComputeDihedralForcePerAtomOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
       const rbmd::Id* atoms_id,
       const rbmd::Id* atom_id_to_idx, const rbmd::Real* dihedral_coeffs_k,
       const rbmd::Id* dihedral_coeffs_sign,
       const rbmd::Id* dihedral_coeffs_multiplicity, const int* d_num_dihedral,
       const int* d_dihedral_type, const rbmd::Id* d_dihedral_atom1,
       const rbmd::Id* d_dihedral_atom2, const rbmd::Id* d_dihedral_atom3,
       const rbmd::Id* d_dihedral_atom4, int dihedral_per_atom,
       const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
       rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
       rbmd::Real* flat_virial, rbmd::Real* energy_dihedral,
       unsigned long long* count_dihedral) {
    unsigned int blocks_per_grid =
        (num_local_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeDihedralForcePerAtom<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, num_local_atoms, num_total_atoms, atoms_id, atom_id_to_idx,
        dihedral_coeffs_k,
        dihedral_coeffs_sign, dihedral_coeffs_multiplicity, d_num_dihedral,
        d_dihedral_type, d_dihedral_atom1, d_dihedral_atom2, d_dihedral_atom3,
        d_dihedral_atom4, dihedral_per_atom, px, py, pz, fx, fy, fz,
        flat_virial, energy_dihedral, count_dihedral));
  }

  void ComputeDihedralOPLSForceOp<device::DEVICE_GPU>::operator()(
       Box box,const rbmd::Id num_atoms,const rbmd::Id num_dihedrals,
       const rbmd::Id* atom_id_to_idx,const rbmd::Real* dihedral_coeffs_k1,
      const rbmd::Real* dihedral_coeffs_k2,const rbmd::Real* dihedral_coeffs_k3,
       const rbmd::Real* dihedral_coeffs_k4, const rbmd::Id* dihedral_type,
       const rbmd::Id* dihedrallisti, const rbmd::Id* dihedrallistj,
      const rbmd::Id* dihedrallistk, const rbmd::Id* dihedrallistw,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,rbmd::Real* flat_virial,
      rbmd::Real* global_virial,rbmd::Real* energy_dihedral) {
    unsigned int blocks_per_grid = (num_dihedrals + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeDihedralOPLSForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, num_atoms,num_dihedrals, atom_id_to_idx, dihedral_coeffs_k1,
        dihedral_coeffs_k2, dihedral_coeffs_k3, dihedral_coeffs_k4,dihedral_type,
        dihedrallisti, dihedrallistj, dihedrallistk, dihedrallistw, px, py, pz,
        fx, fy, fz, flat_virial,global_virial,energy_dihedral));
  }

  void ComputeDihedralOPLSForcePerAtomOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
       const rbmd::Id* atoms_id,
       const rbmd::Id* atom_id_to_idx, const rbmd::Real* dihedral_coeffs_k1,
       const rbmd::Real* dihedral_coeffs_k2,
       const rbmd::Real* dihedral_coeffs_k3,
       const rbmd::Real* dihedral_coeffs_k4, const int* d_num_dihedral,
       const int* d_dihedral_type, const rbmd::Id* d_dihedral_atom1,
       const rbmd::Id* d_dihedral_atom2, const rbmd::Id* d_dihedral_atom3,
       const rbmd::Id* d_dihedral_atom4, int dihedral_per_atom,
       const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
       rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
       rbmd::Real* flat_virial, rbmd::Real* energy_dihedral,
       unsigned long long* count_dihedral) {
    unsigned int blocks_per_grid =
        (num_local_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(
        ComputeDihedralOPLSForcePerAtom<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
            box, num_local_atoms, num_total_atoms, atoms_id, atom_id_to_idx,
            dihedral_coeffs_k1, dihedral_coeffs_k2, dihedral_coeffs_k3,
            dihedral_coeffs_k4, d_num_dihedral, d_dihedral_type,
            d_dihedral_atom1, d_dihedral_atom2, d_dihedral_atom3,
            d_dihedral_atom4, dihedral_per_atom, px, py, pz, fx, fy, fz,
            flat_virial, energy_dihedral, count_dihedral));
  }

void ComputeDihedralFourierForceOp<device::DEVICE_GPU>::operator()(
  Box box, const rbmd::Id num_atoms, const rbmd::Id num_dihedrals,
  const rbmd::Id* atom_id_to_idx,
  const rbmd::Id* nterms, const rbmd::Id* fourier_offsets,
  const rbmd::Real* fourier_k, const rbmd::Id* fourier_multiplicity,
  const rbmd::Real* fourier_cos_shift, const rbmd::Real* fourier_sin_shift,
  const rbmd::Id* dihedral_type, const rbmd::Id* dihedrallisti,
  const rbmd::Id* dihedrallistj, const rbmd::Id* dihedrallistk,
  const rbmd::Id* dihedrallistw, const rbmd::Real* px, const rbmd::Real* py,
  const rbmd::Real* pz,rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
  rbmd::Real* flat_virial, rbmd::Real* global_virial,
  rbmd::Real* energy_dihedral) {
  unsigned int blocks_per_grid = (num_dihedrals + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(ComputeDihedralFourierForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      box, num_atoms,num_dihedrals, atom_id_to_idx, nterms,fourier_offsets,fourier_k,
      fourier_multiplicity,fourier_cos_shift, fourier_sin_shift, dihedral_type,
      dihedrallisti, dihedrallistj, dihedrallistk, dihedrallistw, px, py, pz,
      fx, fy, fz, flat_virial,global_virial,energy_dihedral));
}

void ComputeDihedralFourierForcePerAtomOp<device::DEVICE_GPU>::operator()(
  Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
  const rbmd::Id* atoms_id, const rbmd::Id* atom_id_to_idx,
  const rbmd::Id* nterms, const rbmd::Id* fourier_offsets,
  const rbmd::Real* fourier_k, const rbmd::Id* fourier_multiplicity,
  const rbmd::Real* fourier_cos_shift, const rbmd::Real* fourier_sin_shift,
  const int* d_num_dihedral, const int* d_dihedral_type,
  const rbmd::Id* d_dihedral_atom1, const rbmd::Id* d_dihedral_atom2,
  const rbmd::Id* d_dihedral_atom3, const rbmd::Id* d_dihedral_atom4,
  int dihedral_per_atom, const rbmd::Real* px, const rbmd::Real* py,
  const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
  rbmd::Real* flat_virial, rbmd::Real* energy_dihedral,
  unsigned long long* count_dihedral) {
  unsigned int blocks_per_grid =
      (num_local_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(
      ComputeDihedralFourierForcePerAtom<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          box, num_local_atoms, num_total_atoms, atoms_id, atom_id_to_idx,
          nterms, fourier_offsets, fourier_k, fourier_multiplicity,
          fourier_cos_shift, fourier_sin_shift, d_num_dihedral, d_dihedral_type,
          d_dihedral_atom1, d_dihedral_atom2, d_dihedral_atom3,
          d_dihedral_atom4, dihedral_per_atom, px, py, pz, fx, fy, fz,
          flat_virial, energy_dihedral, count_dihedral));
}

  // force of improper
  void ComputeImproperHarmonicForceOp<device::DEVICE_GPU>::operator()(
    Box box,const rbmd::Id num_atoms,const rbmd::Id num_impropers,
    const rbmd::Id* atom_id_to_idx,const rbmd::Real* improper_coeffs_k,
    const rbmd::Real* improper_coeffs_chi,const rbmd::Id* improper_type,
    const rbmd::Id* improperlisti,const rbmd::Id* improperlistj,
    const rbmd::Id* improperlistk,const rbmd::Id* improperlistw,
    const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
    rbmd::Real* fx,rbmd::Real* fy,rbmd::Real* fz,
    rbmd::Real* flat_virial,rbmd::Real* energy_improper) {
      unsigned int blocks_per_grid = (num_impropers + BLOCK_SIZE - 1) / BLOCK_SIZE;

      CHECK_KERNEL(ComputeImproperHarmonicForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          box,num_atoms,num_impropers, atom_id_to_idx, improper_coeffs_k,
          improper_coeffs_chi, improper_type,improperlisti, improperlistj,
          improperlistk, improperlistw, px, py, pz,fx, fy, fz,
          flat_virial,energy_improper));
  }

void ComputeImproperCVFFForceOp<device::DEVICE_GPU>::operator()(
  Box box,const rbmd::Id num_atoms,const rbmd::Id num_impropers,
  const rbmd::Id* atom_id_to_idx,const rbmd::Real* improper_coeffs_k,
  const rbmd::Id* improper_coeffs_d, const rbmd::Id* improper_coeffs_n,
  const rbmd::Id* improper_type,
  const rbmd::Id* improperlisti,const rbmd::Id* improperlistj,
  const rbmd::Id* improperlistk,const rbmd::Id* improperlistw,
  const rbmd::Real* px,const rbmd::Real* py,const rbmd::Real* pz,
  rbmd::Real* fx,rbmd::Real* fy,rbmd::Real* fz,
  rbmd::Real* flat_virial,rbmd::Real* energy_improper) {
  unsigned int blocks_per_grid = (num_impropers + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(ComputeImproperCVFFForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      box,num_atoms,num_impropers, atom_id_to_idx, improper_coeffs_k,
      improper_coeffs_d,improper_coeffs_n,
      improper_type,improperlisti, improperlistj,
      improperlistk, improperlistw, px, py, pz,fx, fy, fz,
      flat_virial,energy_improper));
}

void ComputeImproperHarmonicPerAtomOp<device::DEVICE_GPU>::operator()(
  Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
  const rbmd::Id* atoms_id,
  const rbmd::Id* atom_id_to_idx, const rbmd::Real* improper_coeffs_k,
  const rbmd::Real* improper_coeffs_chi, const int* d_num_improper,
  const int* d_improper_type, const rbmd::Id* d_improper_atom1,
  const rbmd::Id* d_improper_atom2, const rbmd::Id* d_improper_atom3,
  const rbmd::Id* d_improper_atom4, int improper_per_atom,
  const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
  rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
  rbmd::Real* flat_virial,
  rbmd::Real* energy_improper, unsigned long long* count_improper) {
  unsigned int blocks_per_grid =
      (num_local_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(
      ComputeImproperHarmonicForcePerAtom<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          box, num_local_atoms, num_total_atoms, atoms_id, atom_id_to_idx,
          improper_coeffs_k,
          improper_coeffs_chi, d_num_improper, d_improper_type, d_improper_atom1,
          d_improper_atom2, d_improper_atom3, d_improper_atom4, improper_per_atom,
          px, py, pz, fx, fy, fz, flat_virial, energy_improper, count_improper));
}

void ComputeImproperCVFFPerAtomOp<device::DEVICE_GPU>::operator()(
  Box box, const rbmd::Id num_local_atoms, const rbmd::Id num_total_atoms,
  const rbmd::Id* atoms_id,
  const rbmd::Id* atom_id_to_idx, const rbmd::Real* improper_coeffs_k,
  const rbmd::Id* improper_coeffs_d, const rbmd::Id* improper_coeffs_n,
  const int* d_num_improper, const int* d_improper_type,
  const rbmd::Id* d_improper_atom1, const rbmd::Id* d_improper_atom2,
  const rbmd::Id* d_improper_atom3, const rbmd::Id* d_improper_atom4,
  int improper_per_atom, const rbmd::Real* px, const rbmd::Real* py,
  const rbmd::Real* pz, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
  rbmd::Real* flat_virial,
  rbmd::Real* energy_improper, unsigned long long* count_improper) {
  unsigned int blocks_per_grid =
      (num_local_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(
      ComputeImproperCVFFForcePerAtom<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          box, num_local_atoms, num_total_atoms, atoms_id, atom_id_to_idx,
          improper_coeffs_k,
          improper_coeffs_d, improper_coeffs_n, d_num_improper, d_improper_type,
          d_improper_atom1, d_improper_atom2, d_improper_atom3, d_improper_atom4,
          improper_per_atom, px, py, pz, fx, fy, fz, flat_virial, energy_improper,
          count_improper));
}

void ReduceVirialOp<device::DEVICE_GPU>::operator()(
const rbmd::Id num_atoms,const rbmd::Id pitch,
const rbmd::Real* d_flat_virial_atom,rbmd::Real* d_virial) {

  const int num_components = 6;  //
  const int grid_size = num_components;  //
  const size_t shared_mem_size = BLOCK_SIZE * sizeof(rbmd::Real);

   CHECK_KERNEL(reduce_virial_kernel<<<grid_size, BLOCK_SIZE, shared_mem_size>>>(
      num_atoms,pitch ,d_flat_virial_atom,d_virial));
}


void ComputeSpecialLJCutCoulForceUserOp<device::DEVICE_GPU>::operator()(
  Box box, const rbmd::Real cut_off, const rbmd::Id num_atoms,const  rbmd::Real qqr2e,
  const rbmd::Real rbsog_sigma,const rbmd::Real rbsog_b, const rbmd::Id rbsog_mmax,
  const rbmd::Real rbsog_w0,const rbmd::Real* taylor_coeff,
  const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
  const rbmd::Real* sigma, const rbmd::Real* eps,
  const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
  const int* nspecial, const rbmd::Id* special, int maxspecial,
  rbmd::Real w12, rbmd::Real w13, rbmd::Real w14,
  const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
  rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
  rbmd::Real* flat_virial, rbmd::Real* total_evdwl, rbmd::Real* total_ecoul)
{
  unsigned int blocks = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CHECK_KERNEL(ComputeSpecialLJCutCoulForceUserKernel<<<blocks, BLOCK_SIZE, 0, 0>>>(
      box, cut_off, num_atoms, qqr2e,rbsog_sigma,
      rbsog_b, rbsog_mmax, rbsog_w0,taylor_coeff,
      atoms_type, atoms_id, sigma, eps, start_id, end_id, id_verletlist,
      nspecial, special, maxspecial, w12, w13, w14,
      charge, px, py, pz, fx, fy, fz, flat_virial, total_evdwl, total_ecoul
  ));
}

}
