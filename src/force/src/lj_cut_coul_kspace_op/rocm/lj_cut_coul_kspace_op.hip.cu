#include "../common/rbmd_define.h"
#include <type_traits>
#include "../lj_op/rocm/lj_op.hip.cu"
#include "lj_cut_coul_kspace_op.h"
#include "model/box.h"

namespace op {

__device__ __forceinline__ unsigned int LJRblMixCounter32(
    unsigned int value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__device__ __forceinline__ unsigned int LJRblFoldId32(rbmd::Id value) {
  const unsigned long long wide = static_cast<unsigned long long>(value);
  return static_cast<unsigned int>(wide) ^
         static_cast<unsigned int>(wide >> 32);
}

__device__ __forceinline__ rbmd::Id LJRblSystematicStart(
    rbmd::Id atom_id, rbmd::Id sample_epoch, rbmd::Id candidate_count) {
  unsigned int key = LJRblFoldId32(atom_id);
  key ^= LJRblMixCounter32(LJRblFoldId32(sample_epoch) + 0x9e3779b9U);
  const unsigned int random = LJRblMixCounter32(key);
  return static_cast<rbmd::Id>(
      (static_cast<unsigned long long>(random) *
       static_cast<unsigned long long>(candidate_count)) >> 32);
}

  //---------device---------//
  // CoulForce
  inline __device__ void CoulCutForce_fix(
                rbmd::Real cut_off,
                rbmd::Real alpha,
                rbmd::Real qqr2e,
                rbmd::Real charge_i,
                rbmd::Real charge_j,
                rbmd::Real px12,
                rbmd::Real py12,
                rbmd::Real pz12,
                rbmd::Real& force_coul_factor,
                rbmd::Real& energy_coul_factor,
                rbmd::Real& force_coul,
                rbmd::Real& energy_coul)
  {
    const rbmd::Real dis_2 = px12 * px12 + py12 * py12 + pz12 * pz12;
    const rbmd::Real dis = SQRT(dis_2);
    const rbmd::Real dis_3 = POW(dis,3.0);
    const rbmd::Real cut_off_2 = cut_off * cut_off;

    //if (dis_2 < cut_off_2 && dis_2 > EPSILON)
    if (dis_2 < cut_off_2)
    {
      rbmd::Real erfcx = SQRT(alpha) * dis;
      rbmd::Real expx = -alpha * dis_2;
      rbmd::Real erfc_value = ERFC(erfcx);

      rbmd::Real gnear_value = (erfc_value) / dis_2 +
              2 * SQRT(alpha) * EXP(expx) / (SQRT(M_PI) * dis);

      force_coul_factor =  -qqr2e * charge_i * charge_j / dis_3;//+
      force_coul =  -qqr2e * charge_i * charge_j * gnear_value / dis;//+

      energy_coul_factor  = 0.5 * qqr2e * charge_i * charge_j / dis;
      energy_coul = qqr2e * (0.5 * charge_i * charge_j * (erfc_value) / dis);
    }
    else
    {
      force_coul_factor = 0.0;
      force_coul  = 0.0;
      energy_coul_factor = 0.0;
      energy_coul = 0.0;
    }
  }

inline __device__ void LJ126CoulCutForce_fix(
              rbmd::Real cut_off,
              rbmd::Real eps_ij, rbmd::Real sigma_ij,
              rbmd::Real alpha,
              rbmd::Real qqr2e,
              rbmd::Real charge_i,
              rbmd::Real charge_j,
              rbmd::Real px12,
              rbmd::Real py12,
              rbmd::Real pz12,
              rbmd::Real& force_lj,
              rbmd::Real& energy_lj,
              rbmd::Real& force_coul_factor,
              rbmd::Real& energy_coul_factor,
              rbmd::Real& force_coul,
              rbmd::Real& energy_coul)
  {
    const rbmd::Real dis_2 = px12 * px12 + py12 * py12 + pz12 * pz12;
    const rbmd::Real dis = SQRT(dis_2);
    const rbmd::Real dis_3 = POW(dis,3.0);
    const rbmd::Real cut_off_2 = cut_off * cut_off;

    //if (dis_2 < cut_off_2 && dis_2 > EPSILON)
    if (dis_2 < cut_off_2)
    {
      rbmd::Real sigmaij_6 = POW(sigma_ij, 6.0);
      rbmd::Real dis_6 = POW(dis_2, 3.0);
      rbmd::Real sigmaij_dis_6 = sigmaij_6 / dis_6;
      force_lj = -24 * eps_ij * ((2 * sigmaij_dis_6 - 1) * sigmaij_dis_6) / dis_2;//+
      energy_lj =
          0.5 * (4 * eps_ij * (sigmaij_6 / dis_6 - 1) * sigmaij_dis_6);
      //
      rbmd::Real erfcx = SQRT(alpha) * dis;
      rbmd::Real expx = -alpha * dis_2;
      rbmd::Real gnear_value = (1.0 - ERF(erfcx)) / dis_2 +
              2 * SQRT(alpha) * EXP(expx) / (SQRT(M_PI) * dis);

      force_coul_factor =  -qqr2e * charge_i * charge_j / dis_3;//+
      force_coul =  -qqr2e * charge_i * charge_j * gnear_value / dis;//+

      energy_coul_factor  = 0.5 * qqr2e * charge_i * charge_j / dis;
      energy_coul = qqr2e * (0.5 * charge_i * charge_j * (1.0 - ERF(SQRT(alpha) * dis)) / dis);
    }
    else
    {
      force_lj = 0.0;
      energy_lj= 0.0;
      force_coul_factor = 0.0;
      force_coul  = 0.0;
      energy_coul_factor = 0.0;
      energy_coul = 0.0;
    }
  }

inline __device__ void CoulCutForce(rbmd::Real cut_off, rbmd::Real alpha,
                                  rbmd::Real qqr2e,
                                  rbmd::Real charge_i, rbmd::Real charge_j,
                                  rbmd::Real px12, rbmd::Real py12,
                                  rbmd::Real pz12, rbmd::Real& force_coul,
                                  rbmd::Real& energy_coul) {
    const rbmd::Real dis_2 = px12 * px12 + py12 * py12 + pz12 * pz12;
    const rbmd::Real dis = SQRT(dis_2);
    const rbmd::Real cut_off_2 = cut_off * cut_off;

    if (dis_2 < cut_off_2) {
      rbmd::Real erfcx = SQRT(alpha) * dis;
      rbmd::Real expx = -alpha * dis_2;
      rbmd::Real erfc_value = ERFC(erfcx);

      rbmd::Real gnear_value = (erfc_value) / dis_2 +
              2 * SQRT(alpha) * EXP(expx) / (SQRT(M_PI) * dis);
      force_coul = - qqr2e * charge_i * charge_j * gnear_value / dis ;
      energy_coul = qqr2e * (0.5 * charge_i * charge_j *
                       (erfc_value) / dis);
    } else {
      force_coul = 0.0;
      energy_coul = 0.0;
    }
  }

  inline __device__ void CoulCutForce_rs_fix(
  rbmd::Real rs,
  rbmd::Real alpha,
  rbmd::Real qqr2e,
  rbmd::Real charge_i,
  rbmd::Real charge_j,
  rbmd::Real px12,
  rbmd::Real py12,
  rbmd::Real pz12,
  rbmd::Real& coul_force_factor,
  rbmd::Real& force_coul)
  {
    const rbmd::Real dis_2 = px12 * px12 + py12 * py12 + pz12 * pz12;
    const rbmd::Real dis = SQRT(dis_2);
    const rbmd::Real dis_3 = POW(dis,3.0);
    const rbmd::Real rs_2 = rs * rs;

    if (dis_2 < rs_2 && dis_2 > EPSILON)
    {
      rbmd::Real erfcx = SQRT(alpha) * dis;
      rbmd::Real expx = -alpha * dis_2;
      rbmd::Real gnear_value = (1.0 - ERF(erfcx)) / dis_2 +
              2 * SQRT(alpha) * EXP(expx) / (SQRT(M_PI) * dis);
      coul_force_factor = - qqr2e * charge_i * charge_j / dis_3;
      force_coul = - qqr2e * charge_i * charge_j * gnear_value / dis ;
    }
    else force_coul = 0.0;
  }

  inline __device__ void CoulCutForce_rcs_fix(
        rbmd::Real rc,
        rbmd::Real rs,
        rbmd::Id  pice_num,
        rbmd::Real alpha,
        rbmd::Real qqr2e,
        rbmd::Real charge_i,
        rbmd::Real charge_j,
        rbmd::Real px12,
        rbmd::Real py12,
        rbmd::Real pz12,
        rbmd::Real& coul_force_factor,
        rbmd::Real& force_coul)
  {
    const rbmd::Real dis_2 = px12 * px12 + py12 * py12 + pz12 * pz12;
    const rbmd::Real dis = SQRT(dis_2);
    const rbmd::Real dis_3 = POW(dis,3.0);
    const rbmd::Real rc_2 = rc * rc;
    const rbmd::Real rs_2 = rs * rs;

    if (dis_2 < rc_2 && dis_2 > rs_2)
    {
      rbmd::Real erfcx = SQRT(alpha) * dis;
      rbmd::Real expx = -alpha * dis_2;
      rbmd::Real gnear_value = (1.0 - ERF(erfcx)) / dis_2 +
              2 * SQRT(alpha) * EXP(expx) / (SQRT(M_PI) * dis);
      coul_force_factor = - qqr2e * charge_i * charge_j / dis_3;
      force_coul = -pice_num* qqr2e * charge_i * charge_j *
        gnear_value / dis ;
    }
    else force_coul = 0.0;

  }

inline __device__ rbmd::Real ComputeRBLPairForce(
    const rbmd::Real distance_sq, const rbmd::Real eps_ij,
    const rbmd::Real sigma_ij, const rbmd::Real coul_prefactor,
    const rbmd::Real charge_j, const rbmd::Real alpha,
    const rbmd::Real sqrt_alpha,
    const rbmd::Real two_sqrt_alpha_over_sqrt_pi) {
  const rbmd::Real r2inv = 1.0 / distance_sq;

  rbmd::Real force_lj = 0.0;
  if (eps_ij != 0.0) {
    const rbmd::Real r6inv = r2inv * r2inv * r2inv;
    const rbmd::Real sigma_sq = sigma_ij * sigma_ij;
    const rbmd::Real sigma_6 = sigma_sq * sigma_sq * sigma_sq;
    const rbmd::Real sigma_r_6 = sigma_6 * r6inv;
    force_lj =
        -24.0 * eps_ij * ((2.0 * sigma_r_6 - 1.0) * sigma_r_6) *
        r2inv;
  }

  const rbmd::Real distance = SQRT(distance_sq);
  const rbmd::Real rinv = 1.0 / distance;
  const rbmd::Real erfc_value = ERFC(sqrt_alpha * distance);
  const rbmd::Real exp_value = EXP(-alpha * distance_sq);
  const rbmd::Real force_coul =
      coul_prefactor * charge_j *
      (erfc_value * rinv * r2inv +
       two_sqrt_alpha_over_sqrt_pi * exp_value * r2inv);

  return force_lj + force_coul;
}

inline __device__ void CoulCutForceUser(
     const  rbmd::Real cut_off, const rbmd::Real px12, const rbmd::Real py12,const rbmd::Real pz12,
     const rbmd::Real qqr2e, const rbmd::Real charge_i, const rbmd::Real charge_j,
     const rbmd::Real taylor_coeff0,const rbmd::Real taylor_coeff1,const rbmd::Real taylor_coeff2,
     const rbmd::Real taylor_coeff3,const rbmd::Real taylor_coeff4,const rbmd::Real taylor_coeff5,
     const rbmd::Real sigma, const rbmd::Real b, const rbmd::Id m_max, const rbmd::Real w0,
    rbmd::Real& force_coul_full, rbmd::Real& energy_coul_full,
    rbmd::Real& force_coul_short, rbmd::Real& energy_coul_short)
{
    const rbmd::Real dis_2 = px12 * px12 + py12 * py12 + pz12 * pz12;
    const rbmd::Real  r2inv = 1.0 / dis_2;
    const rbmd::Real cut_off_2 = cut_off * cut_off;
    if (dis_2 < cut_off_2) {

      rbmd::Real rinv = SQRT(r2inv);
      rbmd::Real  r3inv = rinv * r2inv;
      rbmd::Real prefactor = qqr2e * charge_i * charge_j;

      // --- 1. Full Coulomb  ---
      force_coul_full = -prefactor * r3inv;
      energy_coul_full = 0.5* prefactor * rinv;

      // --- 2. SOG Short-Range Force (Taylor Expansion) ---
      rbmd::Real poly = taylor_coeff0
                      + taylor_coeff1 * dis_2
                      + taylor_coeff2 * POW(dis_2,2.0)
                      + taylor_coeff3 * POW(dis_2,3.0)
                      + taylor_coeff4 * POW(dis_2,4.0)
                      + taylor_coeff5 * POW(dis_2,5.0);

      force_coul_short = -prefactor * (r3inv + poly);

      // --- 3. SOG Short-Range Energy (Gaussian Sum Loop) ---
      // ecoul = prefactor * (rinv - sume)
      rbmd::Real sume = 0.0;
      rbmd::Real two_sigma2 = 2.0 * sigma * sigma;
      rbmd::Real norm_factor = 2.0 * LOG(b) / SQRT(2.0 * M_PI * sigma * sigma);

      // l = 0 term
      sume += norm_factor * w0 * EXP(-dis_2 / two_sigma2);

      // l > 0 terms
      for (int l = 1; l < m_max; l++) {
        sume += norm_factor *(1.0 / POW(b, l + 0.00)) *
          EXP(-dis_2 / (2 * sigma * sigma * POW(b, l + 0.00) * POW(b, l + 0.00)));
      }

      energy_coul_short = 0.5* prefactor * (rinv - sume);
    }
    else {
      force_coul_full = 0.0;
      energy_coul_full = 0.0;
      force_coul_short = 0.0;
      energy_coul_short =0.0;
    }

}

  //------global---------//
  // verlet-list: LJCoulCutForce
  __global__ void ComputeLJCutCoulForce(
       Box box, const rbmd::Real cut_off,
      const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* atoms_type, const rbmd::Real* sigma, const rbmd::Real* eps,
      const rbmd::Id* start_id, const rbmd::Id* end_id,
      const rbmd::Id* id_verletlist, const rbmd::Real* charge,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,rbmd::Real* flat_virial,
      rbmd::Real* total_evdwl,rbmd::Real* total_ecoul) {
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
      rbmd::Id typei = atoms_type[tid1];
      rbmd::Real eps_i = eps[typei];
      rbmd::Real sigma_i = sigma[typei];
      rbmd::Real charge_i = charge[tid1];
      rbmd::Real x1 = px[tid1];
      rbmd::Real y1 = py[tid1];
      rbmd::Real z1 = pz[tid1];

      for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
        rbmd::Id tid2 = id_verletlist[j];
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
        MinImageDistance(box, px12, py12, pz12);
        rbmd::Real force_lj, force_coul, force_pair;
        rbmd::Real energy_lj, energy_coul;
        // lj cut
        lj126(cut_off, px12, py12, pz12, eps_ij, sigma_ij, force_lj, energy_lj);

        // Coul cut
        CoulCutForce(cut_off, alpha, qqr2e,charge_i, charge_j,
                         px12, py12, pz12, force_coul, energy_coul);

        force_pair = force_lj + force_coul;
        sum_fx += force_pair * px12;
        sum_fy += force_pair * py12;
        sum_fz += force_pair * pz12;
        sum_elj += energy_lj;
        sum_ecoul += energy_coul;

        rbmd::Real local_virial_xx,local_virial_yy,local_virial_zz,
    local_virial_xy,local_virial_xz,local_virial_yz;
        ComputeVirial(px12, py12, pz12,force_pair,local_virial_xx,local_virial_yy,
  local_virial_zz,local_virial_xy,local_virial_xz,local_virial_yz);
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
        flat_virial[ i * num_atoms + tid1 ] = sum_virial[i];
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

__global__ void ComputeLJCutCoulForceUserKernel(
  Box box, const rbmd::Real cut_off, const rbmd::Id num_atoms,const  rbmd::Real qqr2e,
   const rbmd::Real rbsog_sigma,const rbmd::Real rbsog_b, const rbmd::Id rbsog_mmax,
   const rbmd::Real rbsog_w0,const rbmd::Real* taylor_coeff,
   const rbmd::Id* atoms_type,
   const rbmd::Real* sigma, const rbmd::Real* eps,
   const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
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
      rbmd::Id typei = atoms_type[tid1];
      rbmd::Real eps_i = eps[typei];
      rbmd::Real sigma_i = sigma[typei];
      rbmd::Real charge_i = charge[tid1];
      rbmd::Real x1 = px[tid1];
      rbmd::Real y1 = py[tid1];
      rbmd::Real z1 = pz[tid1];

      for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
        rbmd::Id tid2 = id_verletlist[j];
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


        rbmd::Real force_lj, force_pair;
        rbmd::Real energy_lj;
          // lj cut
        lj126(cut_off, x12, y12, z12, eps_ij, sigma_ij,
          force_lj, energy_lj);

          // --- . Coul Calculation (RBSOG) ---
        rbmd::Real f_coul_full, e_coul_full, f_coul_short, e_coul_short;
        CoulCutForceUser(cut_off, x12, y12,z12, qqr2e, charge_i, charge_j,
                           taylor_coeff[0], taylor_coeff[1],
                           taylor_coeff[2],taylor_coeff[3],
                           taylor_coeff[4], taylor_coeff[5],
                           rbsog_sigma, rbsog_b, rbsog_mmax, rbsog_w0,
                           f_coul_full, e_coul_full,
                           f_coul_short, e_coul_short);

        force_pair = force_lj  + f_coul_short;
        // printf("force_pair:  %f\n",force_pair);

         sum_fx += x12 * force_pair;
         sum_fy += y12 * force_pair;
         sum_fz += z12 * force_pair;

         sum_elj +=  energy_lj;
         sum_ecoul += e_coul_short;

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

  // verlet-list: LJCutCoul Energy
  __global__ void ComputeLJCutCoulEnergy(
       Box box, const rbmd::Real cut_off,
      const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* atoms_type, const rbmd::Real* sigma, const rbmd::Real* eps,
      const rbmd::Id* start_id, const rbmd::Id* end_id,
      const rbmd::Id* id_verletlist, const rbmd::Real* charge,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* flat_virial,rbmd::Real* total_evdwl, rbmd::Real* total_ecoul) {
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
      rbmd::Id typei = atoms_type[tid1];
      rbmd::Real eps_i = eps[typei];
      rbmd::Real sigma_i = sigma[typei];
      rbmd::Real charge_i = charge[tid1];
      rbmd::Real x1 = px[tid1];
      rbmd::Real y1 = py[tid1];
      rbmd::Real z1 = pz[tid1];

      for (int j = start_id[tid1]; j < end_id[tid1]; ++j) {
        rbmd::Id tid2 = id_verletlist[j];
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
        MinImageDistance(box, px12, py12, pz12);
        rbmd::Real force_lj, force_coul;
        rbmd::Real energy_lj, energy_coul;
        // lj cut
        lj126(cut_off, px12, py12, pz12, eps_ij, sigma_ij, force_lj, energy_lj);

        // Coul cut
        CoulCutForce(cut_off, alpha, qqr2e, charge_i, charge_j,
                     px12, py12, pz12, force_coul, energy_coul);

        sum_elj += energy_lj;
        sum_ecoul += energy_coul;

        rbmd::Real force_pair =  force_lj+force_coul;
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
        flat_virial[ i * num_atoms + tid1 ] = sum_virial[i];
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


  // RBL: LJCutCoul
  template <bool FusedCandidateSampling>
  __global__ void ComputeLJCutCoulRBLForce(
       Box box, const rbmd::Real rs, const rbmd::Real rc,
      const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
      const rbmd::Id pice_num, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* __restrict__ atoms_type,
      const rbmd::Id* __restrict__ atoms_id,
      const rbmd::Real* __restrict__ sigma,
      const rbmd::Real* __restrict__ eps,
      const rbmd::Id* __restrict__ start_id,
      const rbmd::Id* __restrict__ end_id,
      const rbmd::Id* __restrict__ id_verletlist,
      const rbmd::Id* __restrict__ id_random_neighbor,
      const rbmd::Id* __restrict__ random_neighbor_num,
      const rbmd::Real* __restrict__ random_neighbor_weight,
      bool, const rbmd::Id* __restrict__ candidate_start,
      const rbmd::Id* __restrict__ candidate_end,
      const rbmd::Id* __restrict__ candidate_num,
      const rbmd::Id* __restrict__ candidate_neighbors,
      rbmd::Id candidate_capacity, bool candidate_implicit_offsets,
      rbmd::Id sample_epoch,
      const rbmd::Real* __restrict__ charge,
      const rbmd::Real* __restrict__ px,
      const rbmd::Real* __restrict__ py,
      const rbmd::Real* __restrict__ pz, rbmd::Real* __restrict__ fx,
      rbmd::Real* __restrict__ fy, rbmd::Real* __restrict__ fz,
      rbmd::Real* __restrict__ force_sum) {
    constexpr bool fused_candidate_sampling = FusedCandidateSampling;
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_fx;
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_fy;
    __shared__ typename BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>::TempStorage
        temp_storage_fz;
    rbmd::Real sum_fsx = 0;
    rbmd::Real sum_fsy = 0;
    rbmd::Real sum_fsz = 0;
    rbmd::Real sum_fcsx = 0;
    rbmd::Real sum_fcsy = 0;
    rbmd::Real sum_fcsz = 0;

    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_atoms) {
      const rbmd::Id typei = __ldg(&atoms_type[tid1]);
      const rbmd::Real eps_i = __ldg(&eps[typei]);
      const rbmd::Real sigma_i =
          eps_i != 0.0 ? __ldg(&sigma[typei]) : rbmd::Real(0);
      const rbmd::Real charge_i = __ldg(&charge[tid1]);

      const rbmd::Real x1 = __ldg(&px[tid1]);
      const rbmd::Real y1 = __ldg(&py[tid1]);
      const rbmd::Real z1 = __ldg(&pz[tid1]);
      const rbmd::Real rs_sq = rs * rs;
      const rbmd::Real rc_sq = rc * rc;
      const rbmd::Real sqrt_alpha = SQRT(alpha);
      const rbmd::Real two_sqrt_alpha_over_sqrt_pi =
          2.0 * sqrt_alpha / SQRT(M_PI);
      const rbmd::Real coul_prefactor = -qqr2e * charge_i;

      // rs
      for (rbmd::Id j = __ldg(&start_id[tid1]);
           j < __ldg(&end_id[tid1]); ++j) {
        const rbmd::Id tid2 = __ldg(&id_verletlist[j]);
        const rbmd::Id typej = __ldg(&atoms_type[tid2]);
        const rbmd::Real eps_j = __ldg(&eps[typej]);
        const rbmd::Real charge_j = __ldg(&charge[tid2]);

        // mix
        rbmd::Real eps_ij = 0.0;
        rbmd::Real sigma_ij = 0.0;
        if (eps_i != 0.0 && eps_j != 0.0) {
          eps_ij = SQRT(eps_i * eps_j);
          sigma_ij = (sigma_i + __ldg(&sigma[typej])) * 0.5;
        }
        const rbmd::Real x2 = __ldg(&px[tid2]);
        const rbmd::Real y2 = __ldg(&py[tid2]);
        const rbmd::Real z2 = __ldg(&pz[tid2]);
        rbmd::Real px12 = x2 - x1;
        rbmd::Real py12 = y2 - y1;
        rbmd::Real pz12 = z2 - z1;
        MinImageDistance(box, px12, py12, pz12);

        const rbmd::Real distance_sq =
            px12 * px12 + py12 * py12 + pz12 * pz12;
        if (distance_sq <= rs_sq && distance_sq > EPSILON) {
          const rbmd::Real force_pair = ComputeRBLPairForce(
              distance_sq, eps_ij, sigma_ij, coul_prefactor, charge_j, alpha,
              sqrt_alpha, two_sqrt_alpha_over_sqrt_pi);
          sum_fsx += force_pair * px12;
          sum_fsy += force_pair * py12;
          sum_fsz += force_pair * pz12;
        }
      }

      // rcs
      const rbmd::Id frequency = pice_num > 0 ? pice_num : rbmd::Id(1);
      const rbmd::Id candidate_count = fused_candidate_sampling
          ? (candidate_implicit_offsets
                 ? __ldg(&candidate_num[tid1])
                 : __ldg(&candidate_end[tid1]) -
                       __ldg(&candidate_start[tid1]))
          : rbmd::Id(0);
      const rbmd::Id requested_draw_count = candidate_count > 0
          ? (candidate_count + frequency - 1) / frequency : rbmd::Id(0);
      const rbmd::Id real_random_num = fused_candidate_sampling
          ? MIN(requested_draw_count, neighbor_sample_num)
          : __ldg(&random_neighbor_num[tid1]);
      const rbmd::Real shell_multiplier = fused_candidate_sampling
          ? (real_random_num > 0
                 ? static_cast<rbmd::Real>(candidate_count) /
                       static_cast<rbmd::Real>(real_random_num)
                 : rbmd::Real(0))
          : (random_neighbor_weight
                 ? __ldg(&random_neighbor_weight[tid1])
                 : static_cast<rbmd::Real>(frequency));
      const unsigned long long candidate_begin = fused_candidate_sampling
          ? (candidate_implicit_offsets
                 ? static_cast<unsigned long long>(tid1) *
                       static_cast<unsigned long long>(candidate_capacity)
                 : static_cast<unsigned long long>(
                       __ldg(&candidate_start[tid1])))
          : 0ULL;
      const rbmd::Id systematic_start =
          fused_candidate_sampling && real_random_num > 0
              ? LJRblSystematicStart(__ldg(&atoms_id[tid1]), sample_epoch,
                                     candidate_count)
              : rbmd::Id(0);
      const rbmd::Id systematic_step =
          real_random_num > 0 ? candidate_count / real_random_num : 0;
      const rbmd::Id systematic_remainder =
          real_random_num > 0 ? candidate_count % real_random_num : 0;
      rbmd::Id systematic_offset = 0;
      rbmd::Id systematic_carry = 0;
      for (rbmd::Id jj = 0; jj < real_random_num; ++jj) {
        rbmd::Id tid2;
        if (fused_candidate_sampling) {
          rbmd::Id candidate_offset = systematic_start + systematic_offset;
          if (candidate_offset >= candidate_count) candidate_offset -= candidate_count;
          tid2 = __ldg(&candidate_neighbors[candidate_begin +
              static_cast<unsigned long long>(candidate_offset)]);
          systematic_offset += systematic_step;
          systematic_carry += systematic_remainder;
          if (systematic_carry >= real_random_num) {
            ++systematic_offset;
            systematic_carry -= real_random_num;
          }
        } else {
          const unsigned long random_neighbor_index =
              static_cast<unsigned long>(jj) *
                  static_cast<unsigned long>(num_atoms) +
              static_cast<unsigned long>(tid1);
          tid2 = __ldg(&id_random_neighbor[random_neighbor_index]);
        }
        const rbmd::Id typej = __ldg(&atoms_type[tid2]);
        const rbmd::Real eps_j = __ldg(&eps[typej]);
        const rbmd::Real charge_j = __ldg(&charge[tid2]);

        // mix
        rbmd::Real eps_ij = 0.0;
        rbmd::Real sigma_ij = 0.0;
        if (eps_i != 0.0 && eps_j != 0.0) {
          eps_ij = SQRT(eps_i * eps_j);
          sigma_ij = (sigma_i + __ldg(&sigma[typej])) * 0.5;
        }
        const rbmd::Real x2 = __ldg(&px[tid2]);
        const rbmd::Real y2 = __ldg(&py[tid2]);
        const rbmd::Real z2 = __ldg(&pz[tid2]);
        rbmd::Real px12 = x2 - x1;
        rbmd::Real py12 = y2 - y1;
        rbmd::Real pz12 = z2 - z1;
        MinImageDistance(box, px12, py12, pz12);

        const rbmd::Real distance_sq =
            px12 * px12 + py12 * py12 + pz12 * pz12;
        if (distance_sq < rc_sq && distance_sq > rs_sq) {
          const rbmd::Real force_pair = shell_multiplier *
              ComputeRBLPairForce(distance_sq, eps_ij, sigma_ij,
                                  coul_prefactor, charge_j, alpha, sqrt_alpha,
                                  two_sqrt_alpha_over_sqrt_pi);
          sum_fcsx += force_pair * px12;
          sum_fcsy += force_pair * py12;
          sum_fcsz += force_pair * pz12;
        }
      }

      // total force = fs + fcs
      fx[tid1] = sum_fsx + sum_fcsx;
      fy[tid1] = sum_fsy + sum_fcsy;
      fz[tid1] = sum_fsz + sum_fcsz;
    }
    const rbmd::Real total_fx = sum_fsx + sum_fcsx;
    const rbmd::Real total_fy = sum_fsy + sum_fcsy;
    const rbmd::Real total_fz = sum_fsz + sum_fcsz;
    const rbmd::Real block_sum_fx =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_fx).Sum(total_fx);
    const rbmd::Real block_sum_fy =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_fy).Sum(total_fy);
    const rbmd::Real block_sum_fz =
        BLOCKREDUCE<rbmd::Real, BLOCK_SIZE>(temp_storage_fz).Sum(total_fz);
    if (threadIdx.x == 0) {
      atomicAdd(&force_sum[0], block_sum_fx);
      atomicAdd(&force_sum[1], block_sum_fy);
      atomicAdd(&force_sum[2], block_sum_fz);
    }
  }


  __global__ void AddForce(const rbmd::Id num_atoms, const rbmd::Real* input_fx,
                           const rbmd::Real* input_fy, const rbmd::Real* input_fz,
                           rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz) {
    unsigned int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid1 < num_atoms) {
      fx[tid1] = fx[tid1] + input_fx[tid1];
      fy[tid1] = fy[tid1] + input_fy[tid1];
      fz[tid1] = fz[tid1] + input_fz[tid1];
    }
  }

  /////////////////////
  // verlet-list: LJCutCoulForce
  void LJCutCoulForceOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Real cut_off,
      const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* atoms_type, const rbmd::Real* sigma, const rbmd::Real* eps,
      const rbmd::Id* start_id, const rbmd::Id* end_id,
      const rbmd::Id* id_verletlist, const rbmd::Real* charge,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz, rbmd::Real* flat_virial,
      rbmd::Real* total_evdwl,rbmd::Real* total_ecoul) {
    unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeLJCutCoulForce<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, cut_off, num_atoms, alpha, qqr2e, atoms_type, sigma, eps,
        start_id, end_id, id_verletlist, charge, px, py, pz, fx, fy, fz,
        flat_virial,total_evdwl ,total_ecoul));
  }

  void LJCutCoulForceUserOp<device::DEVICE_GPU>::operator()(
  Box box, const rbmd::Real cut_off, const rbmd::Id num_atoms,const  rbmd::Real qqr2e,
   const rbmd::Real rbsog_sigma,const rbmd::Real rbsog_b, const rbmd::Id rbsog_mmax,
   const rbmd::Real rbsog_w0,const rbmd::Real* taylor_coeff,
   const rbmd::Id* atoms_type,
   const rbmd::Real* sigma, const rbmd::Real* eps,
   const rbmd::Id* start_id, const rbmd::Id* end_id, const rbmd::Id* id_verletlist,
   const rbmd::Real* charge, const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
   rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
   rbmd::Real* flat_virial, rbmd::Real* total_evdwl, rbmd::Real* total_ecoul) {
      unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

      CHECK_KERNEL(ComputeLJCutCoulForceUserKernel<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
          box, cut_off, num_atoms, qqr2e, rbsog_sigma,rbsog_b,rbsog_mmax,rbsog_w0,
          taylor_coeff,atoms_type, sigma, eps,start_id, end_id, id_verletlist, charge,
          px, py, pz, fx, fy, fz,flat_virial,total_evdwl ,total_ecoul));
    }

  // RBL:  LJCutCoul
  void LJCutCoulRBLForceOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Real rs, const rbmd::Real rc,
      const rbmd::Id num_atoms, const rbmd::Id neighbor_sample_num,
      const rbmd::Id pice_num, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* atoms_type, const rbmd::Id* atoms_id,
      const rbmd::Real* sigma, const rbmd::Real* eps,
      const rbmd::Id* start_id, const rbmd::Id* end_id,
      const rbmd::Id* id_verletlist, const rbmd::Id* id_random_neighbor,
      const rbmd::Id* random_neighbor_num,
      const rbmd::Real* random_neighbor_weight,
      bool fused_candidate_sampling, const rbmd::Id* candidate_start,
      const rbmd::Id* candidate_end, const rbmd::Id* candidate_num,
      const rbmd::Id* candidate_neighbors, rbmd::Id candidate_capacity,
      bool candidate_implicit_offsets, rbmd::Id sample_epoch,
      const rbmd::Real* charge,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz,
      rbmd::Real* force_sum) {
    unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto launch_rbl = [&](auto fused_tag) {
      constexpr bool kFused = decltype(fused_tag)::value;
      CHECK_KERNEL((ComputeLJCutCoulRBLForce<kFused>
                    <<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
                        box, rs, rc, num_atoms, neighbor_sample_num,
                        pice_num, alpha, qqr2e, atoms_type, atoms_id, sigma,
                        eps, start_id, end_id, id_verletlist,
                        id_random_neighbor, random_neighbor_num,
                        random_neighbor_weight, fused_candidate_sampling,
                        candidate_start, candidate_end, candidate_num,
                        candidate_neighbors, candidate_capacity,
                        candidate_implicit_offsets, sample_epoch, charge,
                        px, py, pz, fx, fy, fz, force_sum)));
    };
    if (fused_candidate_sampling) {
      launch_rbl(std::true_type{});
    } else {
      launch_rbl(std::false_type{});
    }
  }

  // verlet-list: LJCutCoul energ
  void LJCutCoulEnergyOp<device::DEVICE_GPU>::operator()(
       Box box, const rbmd::Real cut_off,
      const rbmd::Id num_atoms, const rbmd::Real alpha, const rbmd::Real qqr2e,
      const rbmd::Id* atoms_type, const rbmd::Real* sigma, const rbmd::Real* eps,
      const rbmd::Id* start_id, const rbmd::Id* end_id,
      const rbmd::Id* id_verletlist, const rbmd::Real* charge,
      const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
      rbmd::Real*  flat_virial,rbmd::Real* total_evdwl, rbmd::Real* total_ecoul) {
    unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(ComputeLJCutCoulEnergy<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
        box, cut_off, num_atoms, alpha, qqr2e, atoms_type, sigma, eps,
        start_id, end_id, id_verletlist, charge, px, py, pz, flat_virial,total_evdwl,
        total_ecoul));
  }

  void AddForceOp<device::DEVICE_GPU>::operator()(const rbmd::Id num_atoms,
                                                  const rbmd::Real* input_fx,
                                                  const rbmd::Real* input_fy,
                                                  const rbmd::Real* input_fz,
                                                  rbmd::Real* fx, rbmd::Real* fy,
                                                  rbmd::Real* fz) {
    unsigned int blocks_per_grid = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_KERNEL(AddForce <<<blocks_per_grid, BLOCK_SIZE, 0, 0 >>>
                    (num_atoms, input_fx, input_fy, input_fz, fx, fy, fz));
          }

}
