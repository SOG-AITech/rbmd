#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef USE_DOUBLE
#define USE_DOUBLE 0
#endif
#ifndef USE_64BIT_IDS
#define USE_64BIT_IDS 1
#endif

namespace rbmd {
using Real = std::conditional_t<USE_DOUBLE != 0, double, float>;
using Id = std::conditional_t<USE_64BIT_IDS != 0, long long, int>;
using Energy = double;
}  // namespace rbmd

struct Real7 {
  rbmd::Real values[7]{};
  __host__ __device__ rbmd::Real& operator[](int index) {
    return values[index];
  }
  __host__ __device__ const rbmd::Real& operator[](int index) const {
    return values[index];
  }
};

struct EAMParameters {
  rbmd::Real drho, dr, rhomax, rhomin;
  rbmd::Id nrho, nr;
};

struct Box {
  rbmd::Real length[3]{rbmd::Real(1000), rbmd::Real(1000), rbmd::Real(1000)};
};

struct LinkedCellDeviceDataPtr {
  rbmd::Id _d_per_dimension_cells[3]{1, 1, 1};
  rbmd::Id _d_nativate_cells_dims[3]{1, 1, 1};
  rbmd::Id _d_nativate_cells_start[3]{0, 0, 0};
  bool _d_covers_whole_domain[3]{true, true, true};
};

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
}  // namespace op

__device__ inline void MinImageDistance(Box box, rbmd::Real& x,
                                        rbmd::Real& y, rbmd::Real& z) {
  x -= box.length[0] * nearbyint(x / box.length[0]);
  y -= box.length[1] * nearbyint(y / box.length[1]);
  z -= box.length[2] * nearbyint(z / box.length[2]);
}

#define BLOCK_SIZE 256
#define BLOCKREDUCE cub::BlockReduce
#define SQRT sqrt
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#include "production_eam_direct.cuh"

namespace baseline {
template <bool COMPUTE_PAIR>
__global__ void Fp(
    Box box, EAMParameters parameters, rbmd::Real cutoff,
    rbmd::Id num_atoms, op::EAMDirectTraversalData traversal,
    const Real7* rhor_spline, const Real7* frho_spline,
    const Real7* z2r_spline, const rbmd::Real* px, const rbmd::Real* py,
    const rbmd::Real* pz, rbmd::Real* fp, rbmd::Energy* embedding_blocks,
    rbmd::Energy* pair_blocks) {
  __shared__ typename cub::BlockReduce<rbmd::Energy, BLOCK_SIZE>::TempStorage
      embedding_storage;
  __shared__ typename cub::BlockReduce<rbmd::Energy, BLOCK_SIZE>::TempStorage
      pair_storage;
  rbmd::Real embedding = 0;
  rbmd::Energy pair_energy = 0;
  const rbmd::Id atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < num_atoms) {
    rbmd::Real rho = 0;
    const rbmd::Real cutoff_sq = cutoff * cutoff;
    const rbmd::Id cell = traversal.per_atom_cell_id[atom];
    for (rbmd::Id stencil = 0; stencil < traversal.neighbor_cell_num;
         ++stencil) {
      const rbmd::Id neighbor_cell = op::EAMDirectNeighborCell(
          traversal.linked_cell, cell, stencil,
          traversal.cell_count_within_cutoff);
      for (rbmd::Id pos = traversal.cell_atom_start[neighbor_cell];
           pos < traversal.cell_atom_end[neighbor_cell]; ++pos) {
        const rbmd::Id neighbor = traversal.sorted_atom_indices[pos];
        if (neighbor == atom) continue;
        rbmd::Real dx = px[atom] - px[neighbor];
        rbmd::Real dy = py[atom] - py[neighbor];
        rbmd::Real dz = pz[atom] - pz[neighbor];
        MinImageDistance(box, dx, dy, dz);
        const rbmd::Real rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < cutoff_sq) {
          rbmd::Real local_rho = 0;
          op::EAMRho(rsq, parameters, rhor_spline, local_rho);
          rho += local_rho;
          if (COMPUTE_PAIR) {
            rbmd::Real unused_force = 0;
            rbmd::Real phi = 0;
            op::EAMForce(rsq, parameters, rhor_spline, z2r_spline,
                         rbmd::Real(0), rbmd::Real(0), unused_force, phi);
            pair_energy += phi;
          }
        }
      }
    }
    op::EAMFp(parameters, frho_spline, rho, fp[atom], embedding);
  }
  const rbmd::Energy embedding_sum =
      cub::BlockReduce<rbmd::Energy, BLOCK_SIZE>(embedding_storage)
          .Sum(embedding);
  if (threadIdx.x == 0) embedding_blocks[blockIdx.x] = embedding_sum;
  if (COMPUTE_PAIR) {
    const rbmd::Energy pair_sum =
        cub::BlockReduce<rbmd::Energy, BLOCK_SIZE>(pair_storage)
            .Sum(rbmd::Energy(0.5) * pair_energy);
    if (threadIdx.x == 0) pair_blocks[blockIdx.x] = pair_sum;
  }
}

__global__ void Force(
    Box box, EAMParameters parameters, rbmd::Real core,
    rbmd::Real cutoff, rbmd::Id num_atoms, rbmd::Id sample_cap,
    rbmd::Id frequency, op::EAMDirectTraversalData traversal,
    const Real7* rhor_spline, const Real7* z2r_spline,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* fp, rbmd::Real* fx, rbmd::Real* fy, rbmd::Real* fz) {
  const rbmd::Id atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= num_atoms) return;
  rbmd::Real core_fx = 0, core_fy = 0, core_fz = 0;
  rbmd::Real shell_fx = 0, shell_fy = 0, shell_fz = 0;
  rbmd::Id shell_index = 0, sampled = 0;
  const rbmd::Real core_sq = core * core;
  const rbmd::Real cutoff_sq = cutoff * cutoff;
  const rbmd::Id cell = traversal.per_atom_cell_id[atom];
  for (rbmd::Id stencil = 0; stencil < traversal.neighbor_cell_num;
       ++stencil) {
    const rbmd::Id neighbor_cell = op::EAMDirectNeighborCell(
        traversal.linked_cell, cell, stencil,
        traversal.cell_count_within_cutoff);
    for (rbmd::Id pos = traversal.cell_atom_start[neighbor_cell];
         pos < traversal.cell_atom_end[neighbor_cell]; ++pos) {
      const rbmd::Id neighbor = traversal.sorted_atom_indices[pos];
      if (neighbor == atom) continue;
      rbmd::Real dx = px[atom] - px[neighbor];
      rbmd::Real dy = py[atom] - py[neighbor];
      rbmd::Real dz = pz[atom] - pz[neighbor];
      MinImageDistance(box, dx, dy, dz);
      const rbmd::Real rsq = dx * dx + dy * dy + dz * dz;
      rbmd::Real pair_force = 0, phi = 0;
      if (rsq < core_sq) {
        op::EAMForce(rsq, parameters, rhor_spline, z2r_spline, fp[atom],
                     fp[neighbor], pair_force, phi);
        core_fx += pair_force * dx;
        core_fy += pair_force * dy;
        core_fz += pair_force * dz;
      } else if (rsq < cutoff_sq) {
        if (shell_index % frequency == 0 && sampled < sample_cap) {
          op::EAMForce(rsq, parameters, rhor_spline, z2r_spline, fp[atom],
                       fp[neighbor], pair_force, phi);
          shell_fx += pair_force * dx;
          shell_fy += pair_force * dy;
          shell_fz += pair_force * dz;
          ++sampled;
        }
        ++shell_index;
      }
    }
  }
  fx[atom] = core_fx + frequency * shell_fx;
  fy[atom] = core_fy + frequency * shell_fy;
  fz[atom] = core_fz + frequency * shell_fz;
}
}  // namespace baseline

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)),
              "cudaMalloc");
  }
  explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
    CheckCuda(cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* data() { return data_; }
  std::vector<T> host() const {
    std::vector<T> result(size_);
    CheckCuda(cudaMemcpy(result.data(), data_, size_ * sizeof(T),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H");
    return result;
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

template <class T>
void RequireExact(const std::vector<T>& actual, const std::vector<T>& expected,
                  const std::string& label) {
  if (actual.size() != expected.size()) throw std::runtime_error(label + " size");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0) {
      throw std::runtime_error(label + " mismatch at " + std::to_string(i) +
                               ": " + std::to_string(actual[i]) + " vs " +
                               std::to_string(expected[i]));
    }
  }
}

int main() {
  try {
    constexpr rbmd::Id kAtoms = 517;
    constexpr rbmd::Id kCandidates = 21;
    constexpr rbmd::Id kStorage = kAtoms + kCandidates;
    constexpr int kBlocks = (kAtoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<rbmd::Id> cell(kAtoms), starts{0, 5, 12}, ends{5, 12, 21};
    std::vector<rbmd::Id> sorted(kCandidates);
    std::vector<rbmd::Real> x(kStorage), y(kStorage), z(kStorage);
    std::vector<rbmd::Real> fp_input(kStorage, rbmd::Real(1));
    std::vector<rbmd::Real> nonlinear_fp(kStorage);
    for (rbmd::Id i = 0; i < kAtoms; ++i) {
      cell[i] = i % 3;
      x[i] = rbmd::Real(i % 17) * rbmd::Real(0.11);
      y[i] = rbmd::Real((i / 17) % 5) * rbmd::Real(0.07);
      nonlinear_fp[i] = rbmd::Real(0.35) + rbmd::Real(i % 23) * rbmd::Real(0.002);
    }
    for (rbmd::Id i = 0; i < kCandidates; ++i) {
      sorted[i] = kAtoms + i;
      x[kAtoms + i] = rbmd::Real(-5.3) + rbmd::Real(i) * rbmd::Real(0.61);
      y[kAtoms + i] = rbmd::Real(0.25) + rbmd::Real(i % 4) * rbmd::Real(0.13);
      z[kAtoms + i] = rbmd::Real(0.17) + rbmd::Real(i % 3) * rbmd::Real(0.09);
      nonlinear_fp[kAtoms + i] =
          rbmd::Real(0.4) + rbmd::Real(i) * rbmd::Real(0.003);
    }
    std::vector<Real7> rho(64), embedding(64), pair(64);
    std::vector<Real7> nonlinear_rho(64), nonlinear_embedding(64),
        nonlinear_pair(64);
    for (int i = 0; i < 64; ++i) {
      rho[i][6] = rbmd::Real(1);
      embedding[i][2] = rbmd::Real(1);
      embedding[i][5] = rbmd::Real(1);
      embedding[i][6] = rbmd::Real(i - 1);
      pair[i][6] = rbmd::Real(2);
      nonlinear_rho[i][0] = rbmd::Real(0.001) * rbmd::Real(i % 3 + 1);
      nonlinear_rho[i][1] = rbmd::Real(-0.002);
      nonlinear_rho[i][2] = rbmd::Real(0.003);
      nonlinear_rho[i][3] = rbmd::Real(0.0005);
      nonlinear_rho[i][4] = rbmd::Real(-0.001);
      nonlinear_rho[i][5] = rbmd::Real(0.002);
      nonlinear_rho[i][6] = rbmd::Real(0.2) + rbmd::Real(i) * rbmd::Real(0.001);
      nonlinear_embedding[i][0] = rbmd::Real(0.0007);
      nonlinear_embedding[i][1] = rbmd::Real(-0.001);
      nonlinear_embedding[i][2] = rbmd::Real(0.03) + rbmd::Real(i) * rbmd::Real(0.0001);
      nonlinear_embedding[i][3] = rbmd::Real(0.001);
      nonlinear_embedding[i][4] = rbmd::Real(-0.002);
      nonlinear_embedding[i][5] = rbmd::Real(0.01);
      nonlinear_embedding[i][6] = rbmd::Real(0.5) + rbmd::Real(i) * rbmd::Real(0.002);
      nonlinear_pair[i][0] = rbmd::Real(0.001);
      nonlinear_pair[i][1] = rbmd::Real(-0.002);
      nonlinear_pair[i][2] = rbmd::Real(0.003);
      nonlinear_pair[i][3] = rbmd::Real(0.0004);
      nonlinear_pair[i][4] = rbmd::Real(-0.001);
      nonlinear_pair[i][5] = rbmd::Real(0.002);
      nonlinear_pair[i][6] = rbmd::Real(1) + rbmd::Real(i) * rbmd::Real(0.005);
    }
    EAMParameters parameters{rbmd::Real(1), rbmd::Real(1), rbmd::Real(63),
                             rbmd::Real(0), 64, 64};
    LinkedCellDeviceDataPtr linked;
    linked._d_per_dimension_cells[0] = 3;
    linked._d_nativate_cells_dims[0] = 3;

    DeviceBuffer<rbmd::Id> d_cell(cell), d_starts(starts), d_ends(ends),
        d_sorted(sorted);
    DeviceBuffer<rbmd::Real> d_x(x), d_y(y), d_z(z), d_fp_input(fp_input),
        d_nonlinear_fp(nonlinear_fp);
    DeviceBuffer<Real7> d_rho(rho), d_embedding(embedding), d_pair(pair);
    DeviceBuffer<Real7> d_nonlinear_rho(nonlinear_rho),
        d_nonlinear_embedding(nonlinear_embedding),
        d_nonlinear_pair(nonlinear_pair);
    DeviceBuffer<LinkedCellDeviceDataPtr> d_linked(
        std::vector<LinkedCellDeviceDataPtr>{linked});
    op::EAMDirectTraversalData traversal{d_cell.data(), d_starts.data(),
        d_ends.data(), d_sorted.data(), d_linked.data(), 3, 1};
    Box box;
    const dim3 grid(kBlocks), block(BLOCK_SIZE);

    auto check_fp = [&](rbmd::Id atom_count, const Real7* rho_spline,
                        const Real7* embedding_spline,
                        const Real7* pair_spline, const std::string& variant) {
      const int blocks = static_cast<int>(
          (atom_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
      const dim3 fp_grid(blocks);
      const std::string suffix = " " + variant + " atoms=" +
                                 std::to_string(atom_count);
      DeviceBuffer<rbmd::Real> fp_baseline(kStorage), fp_current(kStorage),
          fp_fused(kStorage);
      DeviceBuffer<rbmd::Energy> embed_baseline(blocks), embed_current(blocks),
          embed_fused(blocks), pair_baseline(blocks), pair_fused(blocks),
          pair_canary(std::vector<rbmd::Energy>(blocks, 1234567.25));
      baseline::Fp<true><<<fp_grid, block>>>(
          box, parameters, rbmd::Real(6), atom_count, traversal, rho_spline,
          embedding_spline, pair_spline, d_x.data(), d_y.data(), d_z.data(),
          fp_baseline.data(), embed_baseline.data(), pair_baseline.data());
      op::FpDirect<false><<<fp_grid, block>>>(
          box, parameters, rbmd::Real(6), atom_count, traversal, rho_spline,
          embedding_spline, d_x.data(), d_y.data(), d_z.data(),
          fp_current.data(), embed_current.data(), pair_spline,
          pair_canary.data());
      op::FpDirect<true><<<fp_grid, block>>>(
          box, parameters, rbmd::Real(6), atom_count, traversal, rho_spline,
          embedding_spline, d_x.data(), d_y.data(), d_z.data(),
          fp_fused.data(), embed_fused.data(), pair_spline, pair_fused.data());
      CheckCuda(cudaDeviceSynchronize(), "FpDirect kernels");
      auto baseline_fp = fp_baseline.host(); baseline_fp.resize(atom_count);
      auto current_fp = fp_current.host(); current_fp.resize(atom_count);
      auto fused_fp = fp_fused.host(); fused_fp.resize(atom_count);
      RequireExact(current_fp, baseline_fp, "Fp no-energy" + suffix);
      RequireExact(fused_fp, baseline_fp, "Fp fused" + suffix);
      RequireExact(embed_current.host(), embed_baseline.host(),
                   "embedding no-energy" + suffix);
      RequireExact(embed_fused.host(), embed_baseline.host(),
                   "embedding fused" + suffix);
      RequireExact(pair_fused.host(), pair_baseline.host(),
                   "pair fused" + suffix);
      RequireExact(pair_canary.host(),
                   std::vector<rbmd::Energy>(blocks, 1234567.25),
                   "pair disabled canary" + suffix);
    };
    check_fp(1, d_rho.data(), d_embedding.data(), d_pair.data(), "analytic");
    check_fp(255, d_rho.data(), d_embedding.data(), d_pair.data(), "analytic");
    check_fp(257, d_nonlinear_rho.data(), d_nonlinear_embedding.data(),
             d_nonlinear_pair.data(), "nonlinear");
    check_fp(kAtoms, d_nonlinear_rho.data(), d_nonlinear_embedding.data(),
             d_nonlinear_pair.data(), "nonlinear");

    auto check_force = [&](rbmd::Real core, rbmd::Id frequency,
                           rbmd::Id sample_cap, const Real7* rho_spline,
                           const Real7* pair_spline,
                           const rbmd::Real* force_fp,
                           const std::string& variant) {
      DeviceBuffer<rbmd::Real> bfx(kAtoms), bfy(kAtoms), bfz(kAtoms),
          cfx(kAtoms), cfy(kAtoms), cfz(kAtoms);
      baseline::Force<<<grid, block>>>(
          box, parameters, core, rbmd::Real(6), kAtoms, sample_cap, frequency,
          traversal, rho_spline, pair_spline, d_x.data(), d_y.data(),
          d_z.data(), force_fp, bfx.data(), bfy.data(), bfz.data());
      op::EAMForceRBLDirect<<<grid, block>>>(
          box, parameters, core, rbmd::Real(6), kAtoms, sample_cap, frequency,
          traversal, rho_spline, pair_spline, d_x.data(), d_y.data(),
          d_z.data(), force_fp, cfx.data(), cfy.data(), cfz.data());
      CheckCuda(cudaDeviceSynchronize(), "force kernels");
      RequireExact(cfx.host(), bfx.host(), "force x " + variant);
      RequireExact(cfy.host(), bfy.host(), "force y " + variant);
      RequireExact(cfz.host(), bfz.host(), "force z " + variant);
    };
    check_force(rbmd::Real(2), 1, 99, d_rho.data(), d_pair.data(),
                d_fp_input.data(), "analytic");
    check_force(rbmd::Real(2), 1, 0, d_rho.data(), d_pair.data(),
                d_fp_input.data(), "analytic");
    check_force(rbmd::Real(2), 2, 1, d_rho.data(), d_pair.data(),
                d_fp_input.data(), "analytic");
    check_force(rbmd::Real(2), 2, 4, d_nonlinear_rho.data(),
                d_nonlinear_pair.data(), d_nonlinear_fp.data(), "nonlinear");
    check_force(rbmd::Real(2), 3, 4, d_nonlinear_rho.data(),
                d_nonlinear_pair.data(), d_nonlinear_fp.data(), "nonlinear");
    check_force(rbmd::Real(6), 1, 0, d_nonlinear_rho.data(),
                d_nonlinear_pair.data(), d_nonlinear_fp.data(), "nonlinear");
    std::cout << "PASS EAM Direct GPU baseline/fusion/cell-map "
              << (USE_DOUBLE ? "double" : "float") << ' '
              << (USE_64BIT_IDS ? "id64" : "id32") << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL EAM Direct GPU: " << error.what() << '\n';
    return 1;
  }
}
