#include "rbmd_define.h"
#include "shake_controller_op.h"
#include <math.h>

namespace op {
#define THREADS_PER_BLOCK 256

__device__ rbmd::Real Dot(const Real3& p_1,
                          const Real3& p_2)
    {
        return p_1.x * p_2.x + p_1.y * p_2.y + p_1.z * p_2.z;
    }

    __device__ bool IsNan(const rbmd::Real& value) {
        //return value != value;  // TODO: isnan
        return (isnan(value) != 0);
}

__global__ void ShakeA(const rbmd::Id num_angle,
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
                       rbmd::Id* flag_pz) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_angle) {

        rbmd::Id id_0x = angle_id_vec[tid].x;
        rbmd::Id id_1x = angle_id_vec[tid].y;
        rbmd::Id id_2x = angle_id_vec[tid].z;
        if (id_0x < 0 || id_1x < 0 || id_2x < 0) {
            return;
        }

    rbmd::Id id_0 = atom_id_to_idx[id_0x];
    rbmd::Id id_1 = atom_id_to_idx[id_1x];
    rbmd::Id id_2 = atom_id_to_idx[id_2x];
    if (id_0 < 0 || id_1 < 0 || id_2 < 0) {
        return;
    }
    if (atoms_type[id_0] < 0 || atoms_type[id_1] < 0 || atoms_type[id_2] < 0) {
        return;
    }

    Real3 shake_position_0, shake_position_1, shake_position_2;
    shake_position_0.x = shake_px[id_0] + shake_vx[id_0] * dt + 0.5 * dt * dt * fx[id_0] / mass[atoms_type[id_0]] * fmt2v;
    shake_position_0.y = shake_py[id_0] + shake_vy[id_0] * dt + 0.5 * dt * dt * fy[id_0] / mass[atoms_type[id_0]] * fmt2v;
    shake_position_0.z = shake_pz[id_0] + shake_vz[id_0] * dt + 0.5 * dt * dt * fz[id_0] / mass[atoms_type[id_0]] * fmt2v;

    shake_position_1.x = shake_px[id_1] + shake_vx[id_1] * dt + 0.5 * dt * dt * fx[id_1] / mass[atoms_type[id_1]] * fmt2v;
    shake_position_1.y = shake_py[id_1] + shake_vy[id_1] * dt + 0.5 * dt * dt * fy[id_1] / mass[atoms_type[id_1]] * fmt2v;
    shake_position_1.z = shake_pz[id_1] + shake_vz[id_1] * dt + 0.5 * dt * dt * fz[id_1] / mass[atoms_type[id_1]] * fmt2v;

    shake_position_2.x = shake_px[id_2] + shake_vx[id_2] * dt + 0.5 * dt * dt * fx[id_2] / mass[atoms_type[id_2]] * fmt2v;
    shake_position_2.y = shake_py[id_2] + shake_vy[id_2] * dt + 0.5 * dt * dt * fy[id_2] / mass[atoms_type[id_2]] * fmt2v;
    shake_position_2.z = shake_pz[id_2] + shake_vz[id_2] * dt + 0.5 * dt * dt * fz[id_2] / mass[atoms_type[id_2]] * fmt2v;

        //printf("tid为：%d----输出结果为：%f, %f, %f \n",tid,shake_position_0.x,shake_position_1.x,shake_position_2.x);


    rbmd::Real bond1 = 1.0;
    rbmd::Real bond2 = 1.0;
    rbmd::Real bond12 = SQRT(bond1 * bond1 + bond2 * bond2 - 2.0 *
      bond1 * bond2 * COS((109.4700 / 180.0) * M_PI));
    
    // minimum image
    Real3 r01 = MinImageDistanceVec(shake_px[id_1], shake_py[id_1],
      shake_pz[id_1], shake_px[id_0], shake_py[id_0], shake_pz[id_0], box);
    Real3 r12 = MinImageDistanceVec(shake_px[id_2], shake_py[id_2],
      shake_pz[id_2], shake_px[id_1], shake_py[id_1], shake_pz[id_1], box);
    Real3 r20 = MinImageDistanceVec(shake_px[id_0], shake_py[id_0],
      shake_pz[id_0], shake_px[id_2], shake_py[id_2], shake_pz[id_2], box);

    // s01,s02,s12 = distance vec after unconstrained update, with PBC
    Real3 s10 = MinImageDistanceVec(shake_position_0.x, shake_position_0.y,
      shake_position_0.z, shake_position_1.x, shake_position_1.y, shake_position_1.z, box);
    Real3 s21 = MinImageDistanceVec(shake_position_1.x, shake_position_1.y,
      shake_position_1.z, shake_position_2.x, shake_position_2.y, shake_position_2.z, box);
    Real3 s02 = MinImageDistanceVec(shake_position_2.x, shake_position_2.y,
      shake_position_2.z, shake_position_0.x, shake_position_0.y, shake_position_0.z, box);

    // scalar distances between atoms
    rbmd::Real r01sq = Dot(r01, r01);
    rbmd::Real r02sq = Dot(r20, r20);
    rbmd::Real r12sq = Dot(r12, r12);
    rbmd::Real s01sq = Dot(s10, s10);
    rbmd::Real s02sq = Dot(s02, s02);
    rbmd::Real s12sq = Dot(s21, s21);

    // matrix coeffs and rhs for lamda equations
    rbmd::Real invmass0 = 1 / mass[atoms_type[id_0]];
    rbmd::Real invmass1 = 1 / mass[atoms_type[id_1]];
    rbmd::Real invmass2 = 1 / mass[atoms_type[id_2]];
    rbmd::Real a11 = 2 * (invmass0 + invmass1) * Dot(s10, r01);
    rbmd::Real a12 = -2 * invmass1 * Dot(s10, r12);
    rbmd::Real a13 = -2 * invmass0 * Dot(s10, r20);
    rbmd::Real a21 = -2 * invmass1 * Dot(s21, r01);
    rbmd::Real a22 = 2 * (invmass1 + invmass2) * Dot(s21, r12);
    rbmd::Real a23 = -2 * invmass2 * Dot(s21, r20);
    rbmd::Real a31 = -2 * invmass0 * Dot(s02, r01);
    rbmd::Real a32 = -2 * invmass2 * Dot(s02, r12);
    rbmd::Real a33 = 2 * (invmass0 + invmass2) * (Dot(s02, r20));

      // inverse of matrix

      rbmd::Real determ = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 -
        a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31;
      if (ABS(determ) < 0.0001)
      {
          printf("Shake determinant = 0.0");
      }


      rbmd::Real determinv = 1 / determ;

      rbmd::Real a11inv = determinv * (a22 * a33 - a23 * a32);
      rbmd::Real a12inv = -determinv * (a12 * a33 - a13 * a32);
      rbmd::Real a13inv = determinv * (a12 * a23 - a13 * a22);
      rbmd::Real a21inv = -determinv * (a21 * a33 - a23 * a31);
      rbmd::Real a22inv = determinv * (a11 * a33 - a13 * a31);
      rbmd::Real a23inv = -determinv * (a11 * a23 - a13 * a21);
      rbmd::Real a31inv = determinv * (a21 * a32 - a22 * a31);
      rbmd::Real a32inv = -determinv * (a11 * a32 - a12 * a31);
      rbmd::Real a33inv = determinv * (a11 * a22 - a12 * a21);


      rbmd::Real r0120 = Dot(r01, r20);
      rbmd::Real r0112 = Dot(r01, r12);
      rbmd::Real r2012 = Dot(r20, r12);

      rbmd::Real quad1_0101 = (invmass0 + invmass1) * (invmass0 + invmass1) * r01sq;
      rbmd::Real quad1_1212 = invmass1 * invmass1 * r12sq;
      rbmd::Real quad1_2020 = invmass0 * invmass0 * r02sq;
      rbmd::Real quad1_0120 = -2 * (invmass0 + invmass1) * invmass0 * r0120;
      rbmd::Real quad1_0112 = -2 * (invmass0 + invmass1) * invmass1 * r0112;
      rbmd::Real quad1_2012 = 2 * invmass0 * invmass1 * r2012;

      rbmd::Real quad2_0101 = invmass1 * invmass1 * r01sq;
      rbmd::Real quad2_1212 = (invmass1 + invmass2) * (invmass1 + invmass2) * r12sq;
      rbmd::Real quad2_2020 = invmass2 * invmass2 * r02sq;
      rbmd::Real quad2_0120 = 2 * invmass1 * invmass2 * r0120;
      rbmd::Real quad2_0112 = -2 * (invmass1 + invmass2) * invmass1 * r0112;
      rbmd::Real quad2_2012 = -2 * (invmass1 + invmass2) * invmass2 * r2012;

      rbmd::Real quad3_0101 = invmass0 * invmass0 * r01sq;
      rbmd::Real quad3_1212 = invmass2 * invmass2 * r12sq;
      rbmd::Real quad3_2020 = (invmass0 + invmass2) * (invmass0 + invmass2) * r02sq;
      rbmd::Real quad3_0120 = -2 * (invmass0 + invmass2) * invmass0 * r0120;
      rbmd::Real quad3_0112 = 2 * invmass0 * invmass2 * r0112;
      rbmd::Real quad3_2012 = -2 * (invmass0 + invmass2) * invmass2 * r2012;

        // iterate until converged
        rbmd::Real tolerance = 0.00001;     // original 0.001
        rbmd::Id max_iter = 5000; // original: 100

        rbmd::Real lamda01 = 0.0;
        rbmd::Real lamda20 = 0.0;
        rbmd::Real lamda12 = 0.0;
        rbmd::Id niter = 0;
        rbmd::Id done = 0;
        rbmd::Id flag_overflow = 0;
        rbmd::Real quad1, quad2, quad3, b1, b2, b3, lamda01_new, lamda20_new, lamda12_new;

        while (!done && niter < max_iter)
        {

            quad1 = quad1_0101 * lamda01 * lamda01 + quad1_2020 * lamda20 * lamda20 +
                    quad1_1212 * lamda12 * lamda12 + quad1_0120 * lamda01 * lamda20 +
                    quad1_0112 * lamda01 * lamda12 + quad1_2012 * lamda20 * lamda12;

            quad2 = quad2_0101 * lamda01 * lamda01 + quad2_2020 * lamda20 * lamda20 +
                    quad2_1212 * lamda12 * lamda12 + quad2_0120 * lamda01 * lamda20 +
                    quad2_0112 * lamda01 * lamda12 + quad2_2012 * lamda20 * lamda12;

            quad3 = quad3_0101 * lamda01 * lamda01 + quad3_2020 * lamda20 * lamda20 +
                    quad3_1212 * lamda12 * lamda12 + quad3_0120 * lamda01 * lamda20 +
                    quad3_0112 * lamda01 * lamda12 + quad3_2012 * lamda20 * lamda12;

            b1 = bond1 * bond1 - s01sq - quad1;
            b2 = bond2 * bond2 - s12sq - quad2;
            b3 = bond12 * bond12 - s02sq - quad3;


            lamda01_new = a11inv * b1 + a12inv * b2 + a13inv * b3;
            lamda12_new = a21inv * b1 + a22inv * b2 + a23inv * b3;
            lamda20_new = a31inv * b1 + a32inv * b2 + a33inv * b3;


            done = 1;
            if (ABS(lamda01_new - lamda01) > tolerance)
                done = 0;
            if (ABS(lamda20_new - lamda20) > tolerance)
                done = 0;
            if (ABS(lamda12_new - lamda12) > tolerance)
                done = 0;

            lamda01 = lamda01_new;
            lamda20 = lamda20_new;
            lamda12 = lamda12_new;

            if (IsNan(lamda01) || IsNan(lamda20) || IsNan(lamda12) ||
                ABS(lamda01) > 1e20 || ABS(lamda20) > 1e20 || ABS(lamda12) > 1e20)
            {
                done = 1;
                flag_overflow = 1;
            }
            niter++;
        }

        Real3 position_constraint_i0;
        Real3 position_constraint_i1;
        Real3 position_constraint_i2;

        position_constraint_i0.x = lamda01 * r01.x * invmass0 - lamda20 * r20.x * invmass0;
        position_constraint_i0.y = lamda01 * r01.y * invmass0 - lamda20 * r20.y * invmass0;
        position_constraint_i0.z = lamda01 * r01.z * invmass0 - lamda20 * r20.z * invmass0;

        position_constraint_i1.x = lamda12 * r12.x * invmass1 - lamda01 * r01.x * invmass1;
        position_constraint_i1.y = lamda12 * r12.y * invmass1 - lamda01 * r01.y * invmass1;
        position_constraint_i1.z = lamda12 * r12.z * invmass1 - lamda01 * r01.z * invmass1;

        position_constraint_i2.x = lamda20 * r20.x * invmass2 - lamda12 * r12.x * invmass2;
        position_constraint_i2.y = lamda20 * r20.y * invmass2 - lamda12 * r12.y * invmass2;
        position_constraint_i2.z = lamda20 * r20.z * invmass2 - lamda12 * r12.z * invmass2;


        Real3 velocity_constraint_i0,velocity_constraint_i1, velocity_constraint_i2;

        velocity_constraint_i0.x = position_constraint_i0.x / dt;
        velocity_constraint_i0.y = position_constraint_i0.y / dt;
        velocity_constraint_i0.z = position_constraint_i0.z / dt;

        velocity_constraint_i1.x = position_constraint_i1.x / dt;
        velocity_constraint_i1.y = position_constraint_i1.y / dt;
        velocity_constraint_i1.z = position_constraint_i1.z / dt;

        velocity_constraint_i2.x = position_constraint_i2.x / dt;
        velocity_constraint_i2.y = position_constraint_i2.y / dt;
        velocity_constraint_i2.z = position_constraint_i2.z / dt;

        Real3 shake_velocity_0,shake_velocity_1,shake_velocity_2;
        shake_velocity_0.x = shake_vx[id_0] + 0.5 * dt * fx[id_0]/mass[atoms_type[id_0]] * fmt2v;
        shake_velocity_0.y = shake_vy[id_0] + 0.5 * dt * fy[id_0]/mass[atoms_type[id_0]] * fmt2v;
        shake_velocity_0.z = shake_vz[id_0] + 0.5 * dt * fz[id_0]/mass[atoms_type[id_0]] * fmt2v;

        shake_velocity_1.x = shake_vx[id_1] + 0.5 * dt * fx[id_1]/mass[atoms_type[id_1]] * fmt2v;
        shake_velocity_1.y = shake_vy[id_1] + 0.5 * dt * fy[id_1]/mass[atoms_type[id_1]] * fmt2v;
        shake_velocity_1.z = shake_vz[id_1] + 0.5 * dt * fz[id_1]/mass[atoms_type[id_1]] * fmt2v;

        shake_velocity_2.x = shake_vx[id_2] + 0.5 * dt * fx[id_2]/mass[atoms_type[id_2]] * fmt2v;
        shake_velocity_2.y = shake_vy[id_2] + 0.5 * dt * fy[id_2]/mass[atoms_type[id_2]] * fmt2v;
        shake_velocity_2.z = shake_vz[id_2] + 0.5 * dt * fz[id_2]/mass[atoms_type[id_2]] * fmt2v;

        // velocity
        shake_vx[id_0] = shake_velocity_0.x + velocity_constraint_i0.x;
        shake_vy[id_0] = shake_velocity_0.y + velocity_constraint_i0.y;
        shake_vz[id_0] = shake_velocity_0.z + velocity_constraint_i0.z;

        shake_vx[id_1] = shake_velocity_1.x + velocity_constraint_i1.x;
        shake_vy[id_1] = shake_velocity_1.y + velocity_constraint_i1.y;
        shake_vz[id_1] = shake_velocity_1.z + velocity_constraint_i1.z;

        shake_vx[id_2] = shake_velocity_2.x + velocity_constraint_i2.x;
        shake_vy[id_2] = shake_velocity_2.y + velocity_constraint_i2.y;
        shake_vz[id_2] = shake_velocity_2.z + velocity_constraint_i2.z;

        // position
        rbmd::Real shake_px_0 = shake_position_0.x + position_constraint_i0.x;
        rbmd::Real shake_py_0 = shake_position_0.y + position_constraint_i0.y;
        rbmd::Real shake_pz_0 = shake_position_0.z + position_constraint_i0.z;

        rbmd::Real shake_px_1= shake_position_1.x + position_constraint_i1.x;
        rbmd::Real shake_py_1= shake_position_1.y + position_constraint_i1.y;
        rbmd::Real shake_pz_1 = shake_position_1.z + position_constraint_i1.z;

        rbmd::Real shake_px_2= shake_position_2.x + position_constraint_i2.x;
        rbmd::Real shake_py_2 = shake_position_2.y + position_constraint_i2.y;
        rbmd::Real shake_pz_2 = shake_position_2.z + position_constraint_i2.z;

        // pbc
        Real3 whole_position_pbc_0, whole_position_pbc_1, whole_position_pbc_2;

        // position
        // whole_position_pbc_0.x = shake_px[id_0];
        // whole_position_pbc_0.y = shake_py[id_0];
        // whole_position_pbc_0.z = shake_pz[id_0];
        //
        // whole_position_pbc_1.x = shake_px[id_1];
        // whole_position_pbc_1.y = shake_py[id_1];
        // whole_position_pbc_1.z = shake_pz[id_1];
        //
        // whole_position_pbc_2.x = shake_px[id_2];
        // whole_position_pbc_2.y = shake_py[id_2];
        // whole_position_pbc_2.z = shake_pz[id_2];

      whole_position_pbc_0.x = shake_px_0;
      whole_position_pbc_0.y = shake_py_0;
      whole_position_pbc_0.z = shake_pz_0;

      whole_position_pbc_1.x = shake_px_1;
      whole_position_pbc_1.y = shake_py_1;
      whole_position_pbc_1.z = shake_pz_1;

      whole_position_pbc_2.x = shake_px_2;
      whole_position_pbc_2.y = shake_py_2;
      whole_position_pbc_2.z = shake_pz_2;
        //position_flag
        Id3 whole_pts_flag_pbc_0, whole_pts_flag_pbc_1, whole_pts_flag_pbc_2;

        whole_pts_flag_pbc_0.x = flag_px[id_0];
        whole_pts_flag_pbc_0.y = flag_py[id_0];
        whole_pts_flag_pbc_0.z = flag_pz[id_0];

        whole_pts_flag_pbc_1.x = flag_px[id_1];
        whole_pts_flag_pbc_1.y = flag_py[id_1];
        whole_pts_flag_pbc_1.z = flag_pz[id_1];

        whole_pts_flag_pbc_2.x = flag_px[id_2];
        whole_pts_flag_pbc_2.y = flag_py[id_2];
        whole_pts_flag_pbc_2.z = flag_pz[id_2];

        // anglelist[0]
      ApplyPBC(box,whole_position_pbc_0.x,whole_position_pbc_0.y,
        whole_position_pbc_0.z,whole_pts_flag_pbc_0.x,
        whole_pts_flag_pbc_0.y,whole_pts_flag_pbc_0.z);
      // anglelist1]
      ApplyPBC(box,whole_position_pbc_1.x,whole_position_pbc_1.y,
  whole_position_pbc_1.z,whole_pts_flag_pbc_1.x,
  whole_pts_flag_pbc_1.y,whole_pts_flag_pbc_1.z);
      // anglelist[2]
      ApplyPBC(box,whole_position_pbc_2.x,whole_position_pbc_2.y,
  whole_position_pbc_2.z,whole_pts_flag_pbc_2.x,
  whole_pts_flag_pbc_2.y,whole_pts_flag_pbc_2.z);

      shake_px[id_0] = whole_position_pbc_0.x;
      shake_py[id_0] = whole_position_pbc_0.y;
      shake_pz[id_0] = whole_position_pbc_0.z;

      shake_px[id_1] = whole_position_pbc_1.x;
      shake_py[id_1] = whole_position_pbc_1.y;
      shake_pz[id_1] = whole_position_pbc_1.z;

      shake_px[id_2] = whole_position_pbc_2.x;
      shake_py[id_2] = whole_position_pbc_2.y;
      shake_pz[id_2] = whole_position_pbc_2.z;

      //flag
      flag_px[id_0] = whole_pts_flag_pbc_0.x;
      flag_py[id_0] = whole_pts_flag_pbc_0.y;
      flag_pz[id_0] = whole_pts_flag_pbc_0.z;

      flag_px[id_1] = whole_pts_flag_pbc_1.x;
      flag_py[id_1] = whole_pts_flag_pbc_1.y;
      flag_pz[id_1] = whole_pts_flag_pbc_1.z;

      flag_px[id_2] = whole_pts_flag_pbc_2.x;
      flag_py[id_2] = whole_pts_flag_pbc_2.y;
      flag_pz[id_2] = whole_pts_flag_pbc_2.z;

      // atomicAdd(&shake_px[id_0], whole_position_pbc_0.x);
      // atomicAdd(&shake_py[id_0], whole_position_pbc_0.y);
      // atomicAdd(&shake_pz[id_0], whole_position_pbc_0.z);
      //
      // atomicAdd(&shake_px[id_1], whole_position_pbc_1.x);
      // atomicAdd(&shake_py[id_1], whole_position_pbc_1.y);
      // atomicAdd(&shake_pz[id_1], whole_position_pbc_1.z);
      //
      // atomicAdd(&shake_px[id_2], whole_position_pbc_2.x);
      // atomicAdd(&shake_py[id_2], whole_position_pbc_2.y);
      // atomicAdd(&shake_pz[id_2], whole_position_pbc_2.z);

  }
}

__global__ void ShakeB(const rbmd::Id num_angle,
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
                       const rbmd::Real* fz) {

        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid < num_angle) {
            rbmd::Id id_0x = angle_id_vec[tid].x;
            rbmd::Id id_1x = angle_id_vec[tid].y;
            rbmd::Id id_2x = angle_id_vec[tid].z;
            if (id_0x < 0 || id_1x < 0 || id_2x < 0) {
                return;
            }

            rbmd::Id id_0 = atom_id_to_idx[id_0x];
            rbmd::Id id_1 = atom_id_to_idx[id_1x];
            rbmd::Id id_2 = atom_id_to_idx[id_2x];
            if (id_0 < 0 || id_1 < 0 || id_2 < 0) {
                return;
            }
            if (atoms_type[id_0] < 0 || atoms_type[id_1] < 0 || atoms_type[id_2] < 0) {
                return;
            }

            Real3 shake_velocity_0,shake_velocity_1,shake_velocity_2;
            shake_velocity_0.x = shake_vx[id_0] + 0.5 * dt * fx[id_0]/mass[atoms_type[id_0]] * fmt2v;
            shake_velocity_0.y = shake_vy[id_0] + 0.5 * dt * fy[id_0]/mass[atoms_type[id_0]] * fmt2v;
            shake_velocity_0.z = shake_vz[id_0] + 0.5 * dt * fz[id_0]/mass[atoms_type[id_0]] * fmt2v;

            shake_velocity_1.x = shake_vx[id_1] + 0.5 * dt * fx[id_1]/mass[atoms_type[id_1]] * fmt2v;
            shake_velocity_1.y = shake_vy[id_1] + 0.5 * dt * fy[id_1]/mass[atoms_type[id_1]] * fmt2v;
            shake_velocity_1.z = shake_vz[id_1] + 0.5 * dt * fz[id_1]/mass[atoms_type[id_1]] * fmt2v;

            shake_velocity_2.x = shake_vx[id_2] + 0.5 * dt * fx[id_2]/mass[atoms_type[id_2]] * fmt2v;
            shake_velocity_2.y = shake_vy[id_2] + 0.5 * dt * fy[id_2]/mass[atoms_type[id_2]] * fmt2v;
            shake_velocity_2.z = shake_vz[id_2] + 0.5 * dt * fz[id_2]/mass[atoms_type[id_2]] * fmt2v;

            // minimum image
            Real3 r01 = MinImageDistanceVec(px[id_1], py[id_1],
              pz[id_1], px[id_0], py[id_0], pz[id_0], box);
            Real3 r12 = MinImageDistanceVec(px[id_2], py[id_2],
              pz[id_2], px[id_1], py[id_1], pz[id_1], box);
            Real3 r20 = MinImageDistanceVec(px[id_0], py[id_0],
              pz[id_0], px[id_2], py[id_2], pz[id_2], box);

            //rbmd::Real sv10[3],sv21[3],sv02[3];
            Real3 sv10, sv21,sv02;
            sv10.x = shake_velocity_0.x - shake_velocity_1.x;
            sv10.y = shake_velocity_0.y - shake_velocity_1.y;
            sv10.z = shake_velocity_0.z - shake_velocity_1.z;

            sv21.x = shake_velocity_1.x - shake_velocity_2.x;
            sv21.y = shake_velocity_1.y - shake_velocity_2.y;
            sv21.z = shake_velocity_1.z - shake_velocity_2.z;

            sv02.x = shake_velocity_2.x - shake_velocity_0.x;
            sv02.y = shake_velocity_2.y - shake_velocity_0.y;
            sv02.z = shake_velocity_2.z - shake_velocity_0.z;

            rbmd::Real invmass0 = 1 / mass[atoms_type[id_0]];
            rbmd::Real invmass1 = 1 / mass[atoms_type[id_1]];
            rbmd::Real invmass2 = 1 / mass[atoms_type[id_2]];

            Real3 c, l;
            Real3 a_0, a_1, a_2;

            // setup matrix
            a_0.x = (invmass1 + invmass0) * Dot(r01, r01);
            a_0.y = -invmass1 * Dot(r01, r12);
            a_0.z = (-invmass0) * Dot(r01, r20);
            a_1.x = a_0.y;
            a_1.y = (invmass1 + invmass2) * Dot(r12, r12);
            a_1.z = -(invmass2) * Dot(r20, r12);
            a_2.x = a_0.z;
            a_2.y = a_1.z;
            a_2.z = (invmass0 + invmass2) * Dot(r20, r20);

            // sestup RHS
            c.x = -Dot(sv10, r01);
            c.y = -Dot(sv21, r12);
            c.z = -Dot(sv02, r20);

            Real3 ai_0, ai_1, ai_2;
            rbmd::Real determ, determinv = 0.0;

            // calculate the determinant of the matrix
            determ = a_0.x * a_1.y * a_2.z + a_0.y * a_1.z * a_2.x + a_0.z *
              a_1.x * a_2.y - a_0.x * a_1.z * a_2.y - a_0.y * a_1.x * a_2.z -
                a_0.z * a_1.y * a_2.x;

            // check if matrix is actually invertible
            if (ABS(determ) < 0.0001)
                printf(" Error: Rattle determinant = 0.0 ");

            // calculate the inverse 3x3 matrix: A^(-1) = (ai_jk)
            determinv = 1 / determ;
            ai_0.x =  determinv * (a_1.y * a_2.z - a_1.z * a_2.y);
            ai_0.y = -determinv * (a_0.y * a_2.z - a_0.z * a_2.y);
            ai_0.z =  determinv * (a_0.y * a_1.z - a_0.z * a_1.y);
            ai_1.x = -determinv * (a_1.x * a_2.z - a_1.z * a_2.x);
            ai_1.y =  determinv * (a_0.x * a_2.z - a_0.z * a_2.x);
            ai_1.z = -determinv * (a_0.x * a_1.z - a_0.z * a_1.x);
            ai_2.x =  determinv * (a_1.x * a_2.y - a_1.y * a_2.x);
            ai_2.y = -determinv * (a_0.x * a_2.y - a_0.y * a_2.x);
            ai_2.z =  determinv * (a_0.x * a_1.y - a_0.y * a_1.x);

            // calculate the solution:  (l01, l02, l12)^T = A^(-1) * c
            l.x = 0;
            l.y = 0;
            l.z = 0;

            l.x += ai_0.x * c.x;
            l.x += ai_0.y * c.y;
            l.x += ai_0.z * c.z;

            l.y += ai_1.x * c.x;
            l.y += ai_1.y * c.y;
            l.y += ai_1.z * c.z;

            l.z += ai_2.x * c.x;
            l.z += ai_2.y * c.y;
            l.z += ai_2.z * c.z;

            // [l01,l02,l12]^T = [lamda12,lamda23,lamda31]^T
            Real3 velocity_constraint_i0, velocity_constraint_i1, velocity_constraint_i2;
            velocity_constraint_i0.x = l.x * r01.x * invmass0 - l.z * r20.x * invmass0;
            velocity_constraint_i0.y = l.x * r01.y * invmass0 - l.z * r20.y * invmass0;
            velocity_constraint_i0.z = l.x * r01.z * invmass0 - l.z * r20.z * invmass0;

            velocity_constraint_i1.x = l.y * r12.x * invmass1 - l.x * r01.x * invmass1;
            velocity_constraint_i1.y = l.y * r12.y * invmass1 - l.x * r01.y * invmass1;
            velocity_constraint_i1.z = l.y * r12.z * invmass1 - l.x * r01.z * invmass1;

            velocity_constraint_i2.x = l.z * r20.x * invmass2 - l.y * r12.x * invmass2;
            velocity_constraint_i2.y = l.z * r20.y * invmass2 - l.y * r12.y * invmass2;
            velocity_constraint_i2.z = l.z * r20.z * invmass2 - l.y * r12.z * invmass2;

            shake_vx[id_0] = shake_velocity_0.x + velocity_constraint_i0.x;
            shake_vy[id_0] = shake_velocity_0.y + velocity_constraint_i0.y;
            shake_vz[id_0] = shake_velocity_0.z + velocity_constraint_i0.z;

            shake_vx[id_1] = shake_velocity_1.x + velocity_constraint_i1.x;
            shake_vy[id_1] = shake_velocity_1.y + velocity_constraint_i1.y;
            shake_vz[id_1] = shake_velocity_1.z + velocity_constraint_i1.z;

            shake_vx[id_2] = shake_velocity_2.x + velocity_constraint_i2.x;
            shake_vy[id_2] = shake_velocity_2.y + velocity_constraint_i2.y;
            shake_vz[id_2] = shake_velocity_2.z + velocity_constraint_i2.z;

          // rbmd::Real shake_vx_0 = shake_velocity_0.x + velocity_constraint_i0.x;
          // rbmd::Real shake_vy_0 = shake_velocity_0.y + velocity_constraint_i0.y;
          // rbmd::Real shake_vz_0 = shake_velocity_0.z + velocity_constraint_i0.z;
          //
          // rbmd::Real shake_vx_1 = shake_velocity_1.x + velocity_constraint_i1.x;
          // rbmd::Real shake_vy_1 = shake_velocity_1.y + velocity_constraint_i1.y;
          // rbmd::Real shake_vz_1 = shake_velocity_1.z + velocity_constraint_i1.z;
          //
          // rbmd::Real shake_vx_2 = shake_velocity_2.x + velocity_constraint_i2.x;
          // rbmd::Real shake_vy_2 = shake_velocity_2.y + velocity_constraint_i2.y;
          // rbmd::Real shake_vz_2 = shake_velocity_2.z + velocity_constraint_i2.z;
          // atomicAdd(&shake_vx[id_0], shake_vx_0);
          // atomicAdd(&shake_vy[id_0], shake_vy_0);
          // atomicAdd(&shake_vz[id_0], shake_vz_0);
          //
          // atomicAdd(&shake_vx[id_1], shake_vx_1);
          // atomicAdd(&shake_vy[id_1], shake_vy_1);
          // atomicAdd(&shake_vz[id_1], shake_vz_1);
          //
          // atomicAdd(&shake_vx[id_2], shake_vx_2);
          // atomicAdd(&shake_vy[id_2], shake_vy_2);
          // atomicAdd(&shake_vz[id_2], shake_vz_2);

            Real3 a00, a01, a02, a10, a11, a12, a20, a21, a22, cal_v12t, cal_v23t, cal_v31t;
            a00.x=(invmass1 + invmass0) * r01.x;
            a00.y=(invmass1 + invmass0) * r01.y;
            a00.z=(invmass1 + invmass0) * r01.z;

            a01.x = -invmass1 * r12.x;
            a01.y = -invmass1 * r12.y;
            a01.z = -invmass1 * r12.z;

            a02.x = (-invmass0) * r20.x;
            a02.y = (-invmass0) * r20.y;
            a02.z = (-invmass0) * r20.z;

            a10.x = -invmass1 * r01.x;
            a10.y = -invmass1 * r01.y;
            a10.z = -invmass1 * r01.z;

            a11.x = (invmass1 + invmass2) * r12.x;
            a11.y = (invmass1 + invmass2) * r12.y;
            a11.z = (invmass1 + invmass2) * r12.z;

            a12.x = -(invmass2) * r20.x;
            a12.y = -(invmass2) * r20.y;
            a12.z = -(invmass2) * r20.z;

            a20.x = (-invmass0) * r01.x;
            a20.y = (-invmass0) * r01.y;
            a20.z = (-invmass0) * r01.z;

            a21.x = -(invmass2) * r12.x;
            a21.y = -(invmass2) * r12.y;
            a21.z = -(invmass2) * r12.z;

            a22.x = (invmass0 + invmass2) * r20.x;
            a22.y = (invmass0 + invmass2) * r20.y;
            a22.z = (invmass0 + invmass2) * r20.z;

            cal_v12t.x = sv10.x + (a00.x * l.x + a01.x * l.y + a02.x * l.z);
            cal_v12t.y = sv10.y + (a00.y * l.x + a01.y * l.y + a02.y * l.z);
            cal_v12t.z = sv10.z + (a00.z * l.x + a01.z * l.y + a02.z * l.z);

            cal_v23t.x = sv21.x + (a10.x * l.x + a11.x * l.y + a12.x * l.z);
            cal_v23t.y = sv21.y + (a10.y * l.x + a11.y * l.y + a12.y * l.z);
            cal_v23t.z = sv21.z + (a10.z * l.x + a11.z * l.y + a12.z * l.z);

            cal_v31t.x = sv02.x + (a20.x * l.x + a21.x * l.y + a22.x * l.z);
            cal_v31t.y = sv02.y + (a20.y * l.x + a21.y * l.y + a22.y * l.z);
            cal_v31t.z = sv02.z + (a20.z * l.x + a21.z * l.y + a22.z * l.z);


            rbmd::Real cal_dv1 = Dot(r01, cal_v12t);
            rbmd::Real cal_dv2 = Dot(r20, cal_v31t);
            rbmd::Real cal_dv12 = Dot(r12, cal_v23t);

            Real3 velocity01, velocity20, velocity12;
            velocity01.x = shake_vx[id_1] - shake_vx[id_0];
            velocity01.y = shake_vy[id_1] - shake_vy[id_0];
            velocity01.z = shake_vz[id_1] - shake_vz[id_0];

            velocity20.x = shake_vx[id_0] - shake_vx[id_2];
            velocity20.y = shake_vy[id_0] - shake_vy[id_2];
            velocity20.z = shake_vz[id_0] - shake_vz[id_2];

            velocity12.x = shake_vx[id_2] - shake_vx[id_1];
            velocity12.y = shake_vy[id_2] - shake_vy[id_1];
            velocity12.z = shake_vz[id_2] - shake_vz[id_1];

            rbmd::Real dv1 = Dot(r01, velocity01);
            rbmd::Real dv2 = Dot(r20, velocity20);
            rbmd::Real dv12 = Dot(r12, velocity12);

            if (ABS(dv1) > 0.1 || ABS(dv2) > 0.1 || ABS(dv12) > 0.1)
            {
                printf("i0 = %d, i1 = %d, i2 = %d\n", id_0, id_1, id_2);
                printf("dv1 = %f, dv2 = %f, dv12 = %f\n", dv1, dv2, dv12);
                printf("cal_dv1 = %f, cal_dv2 = %f, cal_dv12 = %f\n", cal_dv1, cal_dv2, cal_dv12);
                printf("velocity_i0 = [%f,%f,%f], velocity_i1 = [%f,%f,%f], velocity_i2 = [%f,%f,%f]\n",
                       shake_vx[id_0], shake_vy[id_0], shake_vz[id_0],
                       shake_vx[id_1], shake_vy[id_1], shake_vz[id_1],
                       shake_vx[id_2], shake_vy[id_2], shake_vz[id_2]);
                printf("\n");
            }
        }
    }

__device__ rbmd::Id LowerBoundGid(const rbmd::Id* sorted_gid,
                                  rbmd::Id count,
                                  rbmd::Id gid) {
  rbmd::Id first = 0;
  rbmd::Id last = count;
  while (first < last) {
    const rbmd::Id mid = first + (last - first) / 2;
    if (sorted_gid[mid] < gid) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  return first;
}

__device__ rbmd::Id UpperBoundGid(const rbmd::Id* sorted_gid,
                                  rbmd::Id count,
                                  rbmd::Id gid) {
  rbmd::Id first = 0;
  rbmd::Id last = count;
  while (first < last) {
    const rbmd::Id mid = first + (last - first) / 2;
    if (sorted_gid[mid] <= gid) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  return first;
}

__device__ rbmd::Id SelectClosestClusterReplica(
    rbmd::Id gid, rbmd::Id anchor_idx, rbmd::Id native_atoms,
    rbmd::Id total_atoms, const rbmd::Id* sorted_gid,
    const rbmd::Id* sorted_idx, const rbmd::Real* px,
    const rbmd::Real* py, const rbmd::Real* pz) {
  const rbmd::Id first = LowerBoundGid(sorted_gid, total_atoms, gid);
  const rbmd::Id last = UpperBoundGid(sorted_gid, total_atoms, gid);
  if (first >= last) {
    return -1;
  }

  // A locally owned replica is canonical. Otherwise select the periodic ghost
  // image closest to the native topology record that emitted this cluster.
  for (rbmd::Id i = first; i < last; ++i) {
    const rbmd::Id candidate = sorted_idx[i];
    if (candidate >= 0 && candidate < native_atoms) {
      return candidate;
    }
  }

  rbmd::Id best = -1;
  rbmd::Real best_distance = 0;
  for (rbmd::Id i = first; i < last; ++i) {
    const rbmd::Id candidate = sorted_idx[i];
    if (candidate < native_atoms || candidate >= total_atoms) {
      continue;
    }
    const rbmd::Real dx = px[candidate] - px[anchor_idx];
    const rbmd::Real dy = py[candidate] - py[anchor_idx];
    const rbmd::Real dz = pz[candidate] - pz[anchor_idx];
    const rbmd::Real distance = dx * dx + dy * dy + dz * dz;
    if (best < 0 || distance < best_distance ||
        (distance == best_distance && candidate < best)) {
      best = candidate;
      best_distance = distance;
    }
  }
  return best;
}

__global__ void BuildShakeClustersKernel(
    rbmd::Id native_atoms, rbmd::Id total_atoms, int angle_per_atom,
    const int* num_angle, const rbmd::Id* angle_atom0,
    const rbmd::Id* angle_atom1, const rbmd::Id* angle_atom2,
    const rbmd::Id* atoms_id, const rbmd::Real* px,
    const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Id* sorted_gid, const rbmd::Id* sorted_idx,
    rbmd::Id* cluster_idx0, rbmd::Id* cluster_idx1,
    rbmd::Id* cluster_idx2, int* cluster_count, int* invalid_count,
    rbmd::Id* first_invalid_gids) {
  const rbmd::Id flat = threadIdx.x +
                        static_cast<rbmd::Id>(blockIdx.x) * blockDim.x;
  const rbmd::Id max_slots =
      native_atoms * static_cast<rbmd::Id>(angle_per_atom);
  if (flat >= max_slots) {
    return;
  }

  const rbmd::Id native_idx = flat / angle_per_atom;
  const int slot = static_cast<int>(flat % angle_per_atom);
  const int count = num_angle[native_idx];
  if (slot >= count || slot < 0) {
    return;
  }

  const rbmd::Id gid0 = angle_atom0[flat];
  const rbmd::Id gid1 = angle_atom1[flat];
  const rbmd::Id gid2 = angle_atom2[flat];
  if (gid0 < 0 || gid1 < 0 || gid2 < 0) {
    return;
  }

  const rbmd::Id native_gid = atoms_id[native_idx];
  if (native_gid != gid0 && native_gid != gid1 && native_gid != gid2) {
    const int prior = atomicAdd(invalid_count, 1);
    if (prior == 0) {
      first_invalid_gids[0] = gid0;
      first_invalid_gids[1] = gid1;
      first_invalid_gids[2] = gid2;
    }
    return;
  }

  const rbmd::Id idx0 = SelectClosestClusterReplica(
      gid0, native_idx, native_atoms, total_atoms, sorted_gid, sorted_idx, px,
      py, pz);
  const rbmd::Id idx1 = SelectClosestClusterReplica(
      gid1, native_idx, native_atoms, total_atoms, sorted_gid, sorted_idx, px,
      py, pz);
  const rbmd::Id idx2 = SelectClosestClusterReplica(
      gid2, native_idx, native_atoms, total_atoms, sorted_gid, sorted_idx, px,
      py, pz);
  if (idx0 < 0 || idx1 < 0 || idx2 < 0) {
    const int prior = atomicAdd(invalid_count, 1);
    if (prior == 0) {
      first_invalid_gids[0] = gid0;
      first_invalid_gids[1] = gid1;
      first_invalid_gids[2] = gid2;
    }
    return;
  }

  rbmd::Id minimum_native_idx = native_atoms;
  if (idx0 < native_atoms) {
    minimum_native_idx = idx0;
  }
  if (idx1 < minimum_native_idx) {
    minimum_native_idx = idx1;
  }
  if (idx2 < minimum_native_idx) {
    minimum_native_idx = idx2;
  }
  if (native_idx != minimum_native_idx) {
    return;
  }

  const int output = atomicAdd(cluster_count, 1);
  cluster_idx0[output] = idx0;
  cluster_idx1[output] = idx1;
  cluster_idx2[output] = idx2;
}

__global__ void ShakeAReplicatedKernel(const rbmd::Id num_cluster,
                                  const rbmd::Id native_atoms,
                                  const rbmd::Real dt,
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
                                  rbmd::Real* shake_dvz) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_cluster) {
    return;
  }

  const rbmd::Id id_0 = cluster_idx0[tid];
  const rbmd::Id id_1 = cluster_idx1[tid];
  const rbmd::Id id_2 = cluster_idx2[tid];
  if (id_0 < 0 || id_1 < 0 || id_2 < 0) {
    return;
  }
  if (atoms_type[id_0] < 0 || atoms_type[id_1] < 0 || atoms_type[id_2] < 0) {
    return;
  }

  const Real3 old0{shake_px[id_0], shake_py[id_0], shake_pz[id_0]};
  const Real3 old1{shake_px[id_1], shake_py[id_1], shake_pz[id_1]};
  const Real3 old2{shake_px[id_2], shake_py[id_2], shake_pz[id_2]};

  const Real3 pred0{px[id_0], py[id_0], pz[id_0]};
  const Real3 pred1{px[id_1], py[id_1], pz[id_1]};
  const Real3 pred2{px[id_2], py[id_2], pz[id_2]};

  const Real3 vel0{vx[id_0], vy[id_0], vz[id_0]};
  const Real3 vel1{vx[id_1], vy[id_1], vz[id_1]};
  const Real3 vel2{vx[id_2], vy[id_2], vz[id_2]};

  const rbmd::Real bond1 = 1.0;
  const rbmd::Real bond2 = 1.0;
  const rbmd::Real bond12 =
      SQRT(bond1 * bond1 + bond2 * bond2 -
           2.0 * bond1 * bond2 * COS((109.4700 / 180.0) * M_PI));

  const Real3 r01 = MinImageDistanceVec(old1.x, old1.y, old1.z,
                                        old0.x, old0.y, old0.z, box);
  const Real3 r12 = MinImageDistanceVec(old2.x, old2.y, old2.z,
                                        old1.x, old1.y, old1.z, box);
  const Real3 r20 = MinImageDistanceVec(old0.x, old0.y, old0.z,
                                        old2.x, old2.y, old2.z, box);

  const Real3 s10 = MinImageDistanceVec(pred0.x, pred0.y, pred0.z,
                                        pred1.x, pred1.y, pred1.z, box);
  const Real3 s21 = MinImageDistanceVec(pred1.x, pred1.y, pred1.z,
                                        pred2.x, pred2.y, pred2.z, box);
  const Real3 s02 = MinImageDistanceVec(pred2.x, pred2.y, pred2.z,
                                        pred0.x, pred0.y, pred0.z, box);

  const rbmd::Real r01sq = Dot(r01, r01);
  const rbmd::Real r02sq = Dot(r20, r20);
  const rbmd::Real r12sq = Dot(r12, r12);
  const rbmd::Real s01sq = Dot(s10, s10);
  const rbmd::Real s02sq = Dot(s02, s02);
  const rbmd::Real s12sq = Dot(s21, s21);

  const rbmd::Real invmass0 = 1 / mass[atoms_type[id_0]];
  const rbmd::Real invmass1 = 1 / mass[atoms_type[id_1]];
  const rbmd::Real invmass2 = 1 / mass[atoms_type[id_2]];
  const rbmd::Real a11 = 2 * (invmass0 + invmass1) * Dot(s10, r01);
  const rbmd::Real a12 = -2 * invmass1 * Dot(s10, r12);
  const rbmd::Real a13 = -2 * invmass0 * Dot(s10, r20);
  const rbmd::Real a21 = -2 * invmass1 * Dot(s21, r01);
  const rbmd::Real a22 = 2 * (invmass1 + invmass2) * Dot(s21, r12);
  const rbmd::Real a23 = -2 * invmass2 * Dot(s21, r20);
  const rbmd::Real a31 = -2 * invmass0 * Dot(s02, r01);
  const rbmd::Real a32 = -2 * invmass2 * Dot(s02, r12);
  const rbmd::Real a33 = 2 * (invmass0 + invmass2) * Dot(s02, r20);

  const rbmd::Real determ = a11 * a22 * a33 + a12 * a23 * a31 +
                            a13 * a21 * a32 - a11 * a23 * a32 -
                            a12 * a21 * a33 - a13 * a22 * a31;
  if (ABS(determ) < 0.0001) {
    return;
  }

  const rbmd::Real determinv = 1 / determ;
  const rbmd::Real a11inv = determinv * (a22 * a33 - a23 * a32);
  const rbmd::Real a12inv = -determinv * (a12 * a33 - a13 * a32);
  const rbmd::Real a13inv = determinv * (a12 * a23 - a13 * a22);
  const rbmd::Real a21inv = -determinv * (a21 * a33 - a23 * a31);
  const rbmd::Real a22inv = determinv * (a11 * a33 - a13 * a31);
  const rbmd::Real a23inv = -determinv * (a11 * a23 - a13 * a21);
  const rbmd::Real a31inv = determinv * (a21 * a32 - a22 * a31);
  const rbmd::Real a32inv = -determinv * (a11 * a32 - a12 * a31);
  const rbmd::Real a33inv = determinv * (a11 * a22 - a12 * a21);

  const rbmd::Real r0120 = Dot(r01, r20);
  const rbmd::Real r0112 = Dot(r01, r12);
  const rbmd::Real r2012 = Dot(r20, r12);

  const rbmd::Real quad1_0101 =
      (invmass0 + invmass1) * (invmass0 + invmass1) * r01sq;
  const rbmd::Real quad1_1212 = invmass1 * invmass1 * r12sq;
  const rbmd::Real quad1_2020 = invmass0 * invmass0 * r02sq;
  const rbmd::Real quad1_0120 = -2 * (invmass0 + invmass1) * invmass0 * r0120;
  const rbmd::Real quad1_0112 = -2 * (invmass0 + invmass1) * invmass1 * r0112;
  const rbmd::Real quad1_2012 = 2 * invmass0 * invmass1 * r2012;

  const rbmd::Real quad2_0101 = invmass1 * invmass1 * r01sq;
  const rbmd::Real quad2_1212 =
      (invmass1 + invmass2) * (invmass1 + invmass2) * r12sq;
  const rbmd::Real quad2_2020 = invmass2 * invmass2 * r02sq;
  const rbmd::Real quad2_0120 = 2 * invmass1 * invmass2 * r0120;
  const rbmd::Real quad2_0112 = -2 * (invmass1 + invmass2) * invmass1 * r0112;
  const rbmd::Real quad2_2012 = -2 * (invmass1 + invmass2) * invmass2 * r2012;

  const rbmd::Real quad3_0101 = invmass0 * invmass0 * r01sq;
  const rbmd::Real quad3_1212 = invmass2 * invmass2 * r12sq;
  const rbmd::Real quad3_2020 =
      (invmass0 + invmass2) * (invmass0 + invmass2) * r02sq;
  const rbmd::Real quad3_0120 = -2 * (invmass0 + invmass2) * invmass0 * r0120;
  const rbmd::Real quad3_0112 = 2 * invmass0 * invmass2 * r0112;
  const rbmd::Real quad3_2012 = -2 * (invmass0 + invmass2) * invmass2 * r2012;

  rbmd::Real lamda01 = 0.0;
  rbmd::Real lamda20 = 0.0;
  rbmd::Real lamda12 = 0.0;
  const rbmd::Real tolerance = 0.00001;
  const rbmd::Id max_iter = 5000;
  for (rbmd::Id iter = 0; iter < max_iter; ++iter) {
    const rbmd::Real quad1 =
        quad1_0101 * lamda01 * lamda01 + quad1_2020 * lamda20 * lamda20 +
        quad1_1212 * lamda12 * lamda12 + quad1_0120 * lamda01 * lamda20 +
        quad1_0112 * lamda01 * lamda12 + quad1_2012 * lamda20 * lamda12;
    const rbmd::Real quad2 =
        quad2_0101 * lamda01 * lamda01 + quad2_2020 * lamda20 * lamda20 +
        quad2_1212 * lamda12 * lamda12 + quad2_0120 * lamda01 * lamda20 +
        quad2_0112 * lamda01 * lamda12 + quad2_2012 * lamda20 * lamda12;
    const rbmd::Real quad3 =
        quad3_0101 * lamda01 * lamda01 + quad3_2020 * lamda20 * lamda20 +
        quad3_1212 * lamda12 * lamda12 + quad3_0120 * lamda01 * lamda20 +
        quad3_0112 * lamda01 * lamda12 + quad3_2012 * lamda20 * lamda12;

    const rbmd::Real b1 = bond1 * bond1 - s01sq - quad1;
    const rbmd::Real b2 = bond2 * bond2 - s12sq - quad2;
    const rbmd::Real b3 = bond12 * bond12 - s02sq - quad3;

    const rbmd::Real lamda01_new = a11inv * b1 + a12inv * b2 + a13inv * b3;
    const rbmd::Real lamda12_new = a21inv * b1 + a22inv * b2 + a23inv * b3;
    const rbmd::Real lamda20_new = a31inv * b1 + a32inv * b2 + a33inv * b3;

    if (ABS(lamda01_new - lamda01) <= tolerance &&
        ABS(lamda20_new - lamda20) <= tolerance &&
        ABS(lamda12_new - lamda12) <= tolerance) {
      lamda01 = lamda01_new;
      lamda20 = lamda20_new;
      lamda12 = lamda12_new;
      break;
    }

    lamda01 = lamda01_new;
    lamda20 = lamda20_new;
    lamda12 = lamda12_new;
    if (IsNan(lamda01) || IsNan(lamda20) || IsNan(lamda12) ||
        ABS(lamda01) > 1e20 || ABS(lamda20) > 1e20 || ABS(lamda12) > 1e20) {
      return;
    }
  }

  const Real3 position_constraint_i0{
      lamda01 * r01.x * invmass0 - lamda20 * r20.x * invmass0,
      lamda01 * r01.y * invmass0 - lamda20 * r20.y * invmass0,
      lamda01 * r01.z * invmass0 - lamda20 * r20.z * invmass0};
  const Real3 position_constraint_i1{
      lamda12 * r12.x * invmass1 - lamda01 * r01.x * invmass1,
      lamda12 * r12.y * invmass1 - lamda01 * r01.y * invmass1,
      lamda12 * r12.z * invmass1 - lamda01 * r01.z * invmass1};
  const Real3 position_constraint_i2{
      lamda20 * r20.x * invmass2 - lamda12 * r12.x * invmass2,
      lamda20 * r20.y * invmass2 - lamda12 * r12.y * invmass2,
      lamda20 * r20.z * invmass2 - lamda12 * r12.z * invmass2};

  const Real3 velocity_constraint_i0{position_constraint_i0.x / dt,
                                     position_constraint_i0.y / dt,
                                     position_constraint_i0.z / dt};
  const Real3 velocity_constraint_i1{position_constraint_i1.x / dt,
                                     position_constraint_i1.y / dt,
                                     position_constraint_i1.z / dt};
  const Real3 velocity_constraint_i2{position_constraint_i2.x / dt,
                                     position_constraint_i2.y / dt,
                                     position_constraint_i2.z / dt};

  if (id_0 < native_atoms) {
    atomicAdd(&shake_dx[id_0], position_constraint_i0.x);
    atomicAdd(&shake_dy[id_0], position_constraint_i0.y);
    atomicAdd(&shake_dz[id_0], position_constraint_i0.z);
    atomicAdd(&shake_dvx[id_0], velocity_constraint_i0.x);
    atomicAdd(&shake_dvy[id_0], velocity_constraint_i0.y);
    atomicAdd(&shake_dvz[id_0], velocity_constraint_i0.z);
  }
  if (id_1 < native_atoms) {
    atomicAdd(&shake_dx[id_1], position_constraint_i1.x);
    atomicAdd(&shake_dy[id_1], position_constraint_i1.y);
    atomicAdd(&shake_dz[id_1], position_constraint_i1.z);
    atomicAdd(&shake_dvx[id_1], velocity_constraint_i1.x);
    atomicAdd(&shake_dvy[id_1], velocity_constraint_i1.y);
    atomicAdd(&shake_dvz[id_1], velocity_constraint_i1.z);
  }
  if (id_2 < native_atoms) {
    atomicAdd(&shake_dx[id_2], position_constraint_i2.x);
    atomicAdd(&shake_dy[id_2], position_constraint_i2.y);
    atomicAdd(&shake_dz[id_2], position_constraint_i2.z);
    atomicAdd(&shake_dvx[id_2], velocity_constraint_i2.x);
    atomicAdd(&shake_dvy[id_2], velocity_constraint_i2.y);
    atomicAdd(&shake_dvz[id_2], velocity_constraint_i2.z);
  }
}

__global__ void ShakeBReplicatedKernel(const rbmd::Id num_cluster,
                                  const rbmd::Id native_atoms,
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
                                  rbmd::Real* shake_dvz) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_cluster) {
    return;
  }

  const rbmd::Id id_0 = cluster_idx0[tid];
  const rbmd::Id id_1 = cluster_idx1[tid];
  const rbmd::Id id_2 = cluster_idx2[tid];
  if (id_0 < 0 || id_1 < 0 || id_2 < 0) {
    return;
  }
  if (atoms_type[id_0] < 0 || atoms_type[id_1] < 0 || atoms_type[id_2] < 0) {
    return;
  }

  const Real3 vel0{vx[id_0], vy[id_0], vz[id_0]};
  const Real3 vel1{vx[id_1], vy[id_1], vz[id_1]};
  const Real3 vel2{vx[id_2], vy[id_2], vz[id_2]};

  const Real3 r01 = MinImageDistanceVec(px[id_1], py[id_1], pz[id_1],
                                        px[id_0], py[id_0], pz[id_0], box);
  const Real3 r12 = MinImageDistanceVec(px[id_2], py[id_2], pz[id_2],
                                        px[id_1], py[id_1], pz[id_1], box);
  const Real3 r20 = MinImageDistanceVec(px[id_0], py[id_0], pz[id_0],
                                        px[id_2], py[id_2], pz[id_2], box);

  const Real3 sv10{vel0.x - vel1.x, vel0.y - vel1.y, vel0.z - vel1.z};
  const Real3 sv21{vel1.x - vel2.x, vel1.y - vel2.y, vel1.z - vel2.z};
  const Real3 sv02{vel2.x - vel0.x, vel2.y - vel0.y, vel2.z - vel0.z};

  const rbmd::Real invmass0 = 1 / mass[atoms_type[id_0]];
  const rbmd::Real invmass1 = 1 / mass[atoms_type[id_1]];
  const rbmd::Real invmass2 = 1 / mass[atoms_type[id_2]];

  Real3 c, l;
  Real3 a_0, a_1, a_2;
  a_0.x = (invmass1 + invmass0) * Dot(r01, r01);
  a_0.y = -invmass1 * Dot(r01, r12);
  a_0.z = (-invmass0) * Dot(r01, r20);
  a_1.x = a_0.y;
  a_1.y = (invmass1 + invmass2) * Dot(r12, r12);
  a_1.z = -(invmass2) * Dot(r20, r12);
  a_2.x = a_0.z;
  a_2.y = a_1.z;
  a_2.z = (invmass0 + invmass2) * Dot(r20, r20);

  c.x = -Dot(sv10, r01);
  c.y = -Dot(sv21, r12);
  c.z = -Dot(sv02, r20);

  Real3 ai_0, ai_1, ai_2;
  const rbmd::Real determ =
      a_0.x * a_1.y * a_2.z + a_0.y * a_1.z * a_2.x + a_0.z * a_1.x * a_2.y -
      a_0.x * a_1.z * a_2.y - a_0.y * a_1.x * a_2.z - a_0.z * a_1.y * a_2.x;
  if (ABS(determ) < 0.0001) {
    return;
  }
  const rbmd::Real determinv = 1 / determ;
  ai_0.x = determinv * (a_1.y * a_2.z - a_1.z * a_2.y);
  ai_0.y = -determinv * (a_0.y * a_2.z - a_0.z * a_2.y);
  ai_0.z = determinv * (a_0.y * a_1.z - a_0.z * a_1.y);
  ai_1.x = -determinv * (a_1.x * a_2.z - a_1.z * a_2.x);
  ai_1.y = determinv * (a_0.x * a_2.z - a_0.z * a_2.x);
  ai_1.z = -determinv * (a_0.x * a_1.z - a_0.z * a_1.x);
  ai_2.x = determinv * (a_1.x * a_2.y - a_1.y * a_2.x);
  ai_2.y = -determinv * (a_0.x * a_2.y - a_0.y * a_2.x);
  ai_2.z = determinv * (a_0.x * a_1.y - a_0.y * a_1.x);

  l.x = ai_0.x * c.x + ai_0.y * c.y + ai_0.z * c.z;
  l.y = ai_1.x * c.x + ai_1.y * c.y + ai_1.z * c.z;
  l.z = ai_2.x * c.x + ai_2.y * c.y + ai_2.z * c.z;

  const Real3 dv0{l.x * r01.x * invmass0 - l.z * r20.x * invmass0,
                  l.x * r01.y * invmass0 - l.z * r20.y * invmass0,
                  l.x * r01.z * invmass0 - l.z * r20.z * invmass0};
  const Real3 dv1{l.y * r12.x * invmass1 - l.x * r01.x * invmass1,
                  l.y * r12.y * invmass1 - l.x * r01.y * invmass1,
                  l.y * r12.z * invmass1 - l.x * r01.z * invmass1};
  const Real3 dv2{l.z * r20.x * invmass2 - l.y * r12.x * invmass2,
                  l.z * r20.y * invmass2 - l.y * r12.y * invmass2,
                  l.z * r20.z * invmass2 - l.y * r12.z * invmass2};

  if (id_0 < native_atoms) {
    atomicAdd(&shake_dvx[id_0], dv0.x);
    atomicAdd(&shake_dvy[id_0], dv0.y);
    atomicAdd(&shake_dvz[id_0], dv0.z);
  }
  if (id_1 < native_atoms) {
    atomicAdd(&shake_dvx[id_1], dv1.x);
    atomicAdd(&shake_dvy[id_1], dv1.y);
    atomicAdd(&shake_dvz[id_1], dv1.z);
  }
  if (id_2 < native_atoms) {
    atomicAdd(&shake_dvx[id_2], dv2.x);
    atomicAdd(&shake_dvy[id_2], dv2.y);
    atomicAdd(&shake_dvz[id_2], dv2.z);
  }
}

__global__ void ApplyShakeCorrectionsKernel(
    const rbmd::Id num_atoms, Box box, const bool apply_position,
    const rbmd::Real* shake_dx, const rbmd::Real* shake_dy,
    const rbmd::Real* shake_dz, const rbmd::Real* shake_dvx,
    const rbmd::Real* shake_dvy, const rbmd::Real* shake_dvz, rbmd::Real* px,
    rbmd::Real* py, rbmd::Real* pz, rbmd::Real* vx, rbmd::Real* vy,
    rbmd::Real* vz, rbmd::Id* flag_px, rbmd::Id* flag_py, rbmd::Id* flag_pz) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_atoms) {
    return;
  }

  if (apply_position) {
    px[tid] += shake_dx[tid];
    py[tid] += shake_dy[tid];
    pz[tid] += shake_dz[tid];
    ApplyPBC(box, px[tid], py[tid], pz[tid], flag_px[tid], flag_py[tid],
             flag_pz[tid]);
  }

  vx[tid] += shake_dvx[tid];
  vy[tid] += shake_dvy[tid];
  vz[tid] += shake_dvz[tid];
}

void ShakeAOp<device::DEVICE_GPU>::operator()(const rbmd::Id num_angle,
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
                                              rbmd::Id* flag_pz) 
{
  unsigned int blocks_per_grid = (num_angle + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(ShakeA<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      num_angle, dt, fmt2v, box, atom_id_to_idx,mass, atoms_type,
      angle_id_vec,shake_px, shake_py, shake_pz, shake_vx, shake_vy,
      shake_vz, fx, fy, fz, flag_px, flag_py, flag_pz));
}

    void ShakeBOp<device::DEVICE_GPU>::operator()(const rbmd::Id num_angle,
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
                                                  const rbmd::Real* fz)
    {
        unsigned int blocks_per_grid = (num_angle + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CHECK_KERNEL(ShakeB<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>
          (num_angle, dt, fmt2v, box, atom_id_to_idx, mass, atoms_type,
           angle_id_vec,px, py, pz, shake_vx, shake_vy, shake_vz, fx, fy, fz));
    }

void BuildShakeClustersOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id native_atoms, const rbmd::Id total_atoms,
    const int angle_per_atom, const int* num_angle,
    const rbmd::Id* angle_atom0, const rbmd::Id* angle_atom1,
    const rbmd::Id* angle_atom2, const rbmd::Id* atoms_id,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Id* sorted_gid, const rbmd::Id* sorted_idx,
    rbmd::Id* cluster_idx0, rbmd::Id* cluster_idx1,
    rbmd::Id* cluster_idx2, int* cluster_count, int* invalid_count,
    rbmd::Id* first_invalid_gids) {
  const rbmd::Id max_slots =
      native_atoms * static_cast<rbmd::Id>(angle_per_atom);
  if (max_slots <= 0) {
    return;
  }
  const unsigned int blocks_per_grid =
      static_cast<unsigned int>((max_slots + BLOCK_SIZE - 1) / BLOCK_SIZE);
  CHECK_KERNEL(BuildShakeClustersKernel<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      native_atoms, total_atoms, angle_per_atom, num_angle, angle_atom0,
      angle_atom1, angle_atom2, atoms_id, px, py, pz, sorted_gid, sorted_idx,
      cluster_idx0, cluster_idx1, cluster_idx2, cluster_count, invalid_count,
      first_invalid_gids));
}

void ShakeAReplicatedOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id num_cluster, const rbmd::Id native_atoms,
    const rbmd::Real dt, Box box,
    const rbmd::Real* mass, const rbmd::Id* atoms_type,
    const rbmd::Id* cluster_idx0,
    const rbmd::Id* cluster_idx1, const rbmd::Id* cluster_idx2,
    const rbmd::Real* shake_px,
    const rbmd::Real* shake_py, const rbmd::Real* shake_pz,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* vx, const rbmd::Real* vy, const rbmd::Real* vz,
    rbmd::Real* shake_dx, rbmd::Real* shake_dy, rbmd::Real* shake_dz,
    rbmd::Real* shake_dvx, rbmd::Real* shake_dvy, rbmd::Real* shake_dvz) {
  const unsigned int blocks_per_grid =
      (num_cluster + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(ShakeAReplicatedKernel<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      num_cluster, native_atoms, dt, box, mass, atoms_type, cluster_idx0,
      cluster_idx1, cluster_idx2, shake_px, shake_py,
      shake_pz, px, py, pz, vx, vy, vz, shake_dx, shake_dy, shake_dz,
      shake_dvx, shake_dvy, shake_dvz));
}

void ShakeBReplicatedOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id num_cluster, const rbmd::Id native_atoms, Box box,
    const rbmd::Real* mass, const rbmd::Id* atoms_type,
    const rbmd::Id* cluster_idx0, const rbmd::Id* cluster_idx1,
    const rbmd::Id* cluster_idx2,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* vx, const rbmd::Real* vy, const rbmd::Real* vz,
    rbmd::Real* shake_dvx, rbmd::Real* shake_dvy, rbmd::Real* shake_dvz) {
  const unsigned int blocks_per_grid =
      (num_cluster + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(ShakeBReplicatedKernel<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      num_cluster, native_atoms, box, mass, atoms_type, cluster_idx0,
      cluster_idx1, cluster_idx2, px, py, pz, vx, vy, vz,
      shake_dvx, shake_dvy, shake_dvz));
}

void ApplyShakeCorrectionsOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id num_atoms, Box box, const bool apply_position,
    const rbmd::Real* shake_dx, const rbmd::Real* shake_dy,
    const rbmd::Real* shake_dz, const rbmd::Real* shake_dvx,
    const rbmd::Real* shake_dvy, const rbmd::Real* shake_dvz, rbmd::Real* px,
    rbmd::Real* py, rbmd::Real* pz, rbmd::Real* vx, rbmd::Real* vy,
    rbmd::Real* vz, rbmd::Id* flag_px, rbmd::Id* flag_py, rbmd::Id* flag_pz) {
  const unsigned int blocks_per_grid =
      (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
  CHECK_KERNEL(ApplyShakeCorrectionsKernel<<<blocks_per_grid, BLOCK_SIZE, 0, 0>>>(
      num_atoms, box, apply_position, shake_dx, shake_dy, shake_dz, shake_dvx,
      shake_dvy, shake_dvz, px, py, pz, vx, vy, vz, flag_px, flag_py,
      flag_pz));
}
}  // namespace op
