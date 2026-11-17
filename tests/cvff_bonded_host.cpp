// Host arithmetic harness for the production kernels extracted by the runner.
// One atom per block makes block sums/atomicAdd ordinary scalar operations.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/types.h"
#include "common/atom_index_map.h"
namespace device { struct DEVICE_GPU {}; }
using STREAM_T = void*;
struct Box { rbmd::Real length = 32; };
struct Dim { unsigned int x = 0; } blockIdx, threadIdx, blockDim{1};
#define __global__
#define __shared__
#define BLOCK_SIZE 1
#define SQRT std::sqrt
#define ACOS std::acos
#define MIN std::min
#define SMALL 0.001
#define CHECK_KERNEL(...) __VA_ARGS__

template <typename T, int N> struct BLOCKREDUCE {
  struct TempStorage {};
  explicit BLOCKREDUCE(TempStorage&) {}
  T Sum(T value) { return value; }
};
template <typename T> void atomicAdd(T* target, T value) { *target += value; }
void MinImageDistance_while(Box box, rbmd::Real& x, rbmd::Real& y,
                            rbmd::Real& z) {
  for (auto* v : {&x, &y, &z})
    *v -= box.length * std::nearbyint(*v / box.length);
}
#include "production_kernels.h"

using Vec = std::array<double, 3>;
using Positions = std::array<Vec, 5>;
const std::array<rbmd::Id, 5> gids{10000000007LL, 2, 10000000019LL, 11, 10000000023LL};
const std::vector<std::array<int, 2>> bonds{{0, 1}, {1, 2}, {2, 3}};
const std::vector<std::array<int, 3>> angles{{0, 1, 2}, {1, 2, 3}};
constexpr double bond_k = 2.5, bond_r0 = 1.25;
constexpr double angle_k = 1.75, angle_theta0 = 110;

Vec displacement(const Positions& p, int a, int b) {
  Vec d{};
  for (int axis = 0; axis < 3; ++axis) {
    d[axis] = p[a][axis] - p[b][axis];
    d[axis] -= 32 * std::nearbyint(d[axis] / 32);
  }
  return d;
}
double dot(Vec a, Vec b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
// Independent scalar potential; all force and tensor checks use derivatives.
double energy(const Positions& p, bool angle) {
  double result = 0;
  if (angle) {
    for (auto ids : angles) {
      const Vec a = displacement(p, ids[0], ids[1]);
      const Vec b = displacement(p, ids[2], ids[1]);
      const double theta = std::acos(std::clamp(dot(a,b) /
          std::sqrt(dot(a,a)*dot(b,b)), -1.0, 1.0));
      const double delta = theta - angle_theta0 * M_PI / 180;
      result += angle_k * delta * delta;
    }
  } else {
    for (auto ids : bonds) {
      Vec d = displacement(p, ids[0], ids[1]);
      const double delta = std::sqrt(dot(d,d)) - bond_r0;
      result += bond_k * delta * delta;
    }
  }
  return result;
}
void check(double actual, double expected, const std::string& label) {
  const double tolerance = sizeof(rbmd::Real) == sizeof(float) ? 8e-5 : 2e-7;
  if (!std::isfinite(actual) || std::abs(actual-expected) >
      tolerance * (1 + std::abs(expected))) {
    throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}
struct Result {
  std::array<Vec, 5> force{};
  std::array<double, 6> virial{};
  double energy = 0;
  unsigned long long count = 0;
};

Result evaluate(const Positions& p, bool angle,
                const std::vector<std::vector<int>>& owners) {
  Result result;
  constexpr int width = 4;
  for (const auto& owned : owners) {
    std::vector<int> order = owned;
    // Deliberately reverse the ghost order: kernels must follow GID mappings.
    for (int i = 4; i >= 0; --i)
      if (std::find(owned.begin(), owned.end(), i) == owned.end()) order.push_back(i);
    const int n = static_cast<int>(owned.size());
    std::vector<rbmd::Id> ids(5), other(5*width, -1),
        ai(5*width, -1), aj(5*width, -1), ak(5*width, -1);
    std::vector<int> counts(5, 0), types(5*width, 0);
    std::vector<rbmd::Real> x(5), y(5), z(5), fx(5, 123), fy(5, 123), fz(5, 123);
    // Guard beyond 6*n catches incorrect total-atoms pitch and ghost writes.
    std::vector<rbmd::Real> virial(6*n+6, 123);
    for (int row = 0; row < 5; ++row) {
      const int atom = order[row];
      ids[row] = gids[atom];
      x[row] = p[atom][0]; y[row] = p[atom][1]; z[row] = p[atom][2];
      if (angle) {
        for (const auto& t : angles) {
          if (std::find(t.begin(), t.end(), atom) == t.end()) continue;
          const int slot = row*width + counts[row]++;
          ai[slot] = gids[t[0]]; aj[slot] = gids[t[1]]; ak[slot] = gids[t[2]];
        }
      } else {
        for (const auto& t : bonds) {
          if (atom != t[0] && atom != t[1]) continue;
          const int slot = row*width + counts[row]++;
          other[slot] = gids[atom == t[0] ? t[1] : t[0]];
        }
      }
    }
    std::vector<rbmd::Id> map_keys(ids);
    std::vector<rbmd::LocalAtomIndex> map_indices(5);
    std::iota(map_indices.begin(), map_indices.end(), 0);
    std::sort(map_indices.begin(), map_indices.end(),
              [&](auto a, auto b) { return ids[a] < ids[b]; });
    for (int i = 0; i < 5; ++i) map_keys[i] = ids[map_indices[i]];
    const rbmd::AtomIndexMapView map{map_keys.data(), map_indices.data(), map_keys.size()};
    rbmd::Real local_energy = 0;
    unsigned long long local_count = 0;
    const rbmd::Real k = angle ? angle_k : bond_k;
    const rbmd::Real equilibrium = angle ? angle_theta0 : bond_r0;
    // Include an out-of-range block, also exercising the empty owned partition.
    for (int row = 0; row <= n; ++row) {
      blockIdx.x = row;
      if (angle) {
        ComputeAngleForcePerAtom(Box{}, n, 5, ids.data(), map, &k,
            &equilibrium, counts.data(), types.data(), ai.data(), aj.data(),
            ak.data(), width, x.data(), y.data(), z.data(), fx.data(), fy.data(),
            fz.data(), virial.data(), &local_energy, &local_count);
      } else {
        ComputeBondForcePerAtom(Box{}, n, 5, ids.data(), map, map.count,
            &k, 1, &equilibrium, counts.data(), types.data(), other.data(), width,
            x.data(), y.data(), z.data(), fx.data(), fy.data(), fz.data(),
            virial.data(), &local_energy, &local_count);
      }
    }
    result.energy += local_energy;
    result.count += local_count;
    for (int row = 0; row < n; ++row) {
      result.force[order[row]] = {fx[row], fy[row], fz[row]};
      for (int c = 0; c < 6; ++c) result.virial[c] += virial[c*n+row];
    }
    for (int row = n; row < 5; ++row) {
      check(fx[row], 123, "ghost force x overwritten");
      check(fy[row], 123, "ghost force y overwritten");
      check(fz[row], 123, "ghost force z overwritten");
    }
    for (std::size_t i = 6*n; i < virial.size(); ++i)
      check(virial[i], 123, "virial guard overwritten");
  }
  return result;
}

int main() {
  try {
    const Positions p{{{1, 2, 3}, {2.5, 2.25, 3.5}, {3.25, 3.5, 3.75},
                       {4.5, 3.75, 5}, {8, 8, 8}}};
    const std::vector<std::vector<std::vector<int>>> partitions{
        {{0,1,2,3,4}}, {{3,1},{4,2,0},{}}, {{0},{1},{2},{3},{4},{}}};
    const std::array<std::array<int,2>,6> tensor_axes{{{0,0},{1,1},{2,2},{0,1},{0,2},{1,2}}};
    constexpr double h = 1e-5;
    for (bool angle : {false, true}) {
      Result reference;
      reference.energy = energy(p, angle);
      reference.count = angle ? angles.size() : bonds.size();
      for (int atom = 0; atom < 5; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
          Positions plus = p, minus = p;
          plus[atom][axis] += h; minus[atom][axis] -= h;
          reference.force[atom][axis] = -(energy(plus,angle)-energy(minus,angle))/(2*h);
        }
      }
      for (int c = 0; c < 6; ++c) {
        const int a = tensor_axes[c][0], b = tensor_axes[c][1];
        Positions plus = p, minus = p;
        for (int atom = 0; atom < 5; ++atom) {
          plus[atom][b] += h*p[atom][a]; minus[atom][b] -= h*p[atom][a];
        }
        reference.virial[c] = -(energy(plus,angle)-energy(minus,angle))/(2*h);
      }
      for (const auto& ownership : partitions) {
        for (bool periodic : {false, true}) {
          Positions shifted = p;
          if (periodic) {
            shifted[0][0] += 32; shifted[1][1] -= 32; shifted[3][2] += 32;
          }
          const Result actual = evaluate(shifted, angle, ownership);
          check(actual.energy, reference.energy, "energy");
          check(actual.count, reference.count, "topology count");
          for (int atom = 0; atom < 5; ++atom)
            for (int axis = 0; axis < 3; ++axis)
              check(actual.force[atom][axis], reference.force[atom][axis], "force derivative");
          for (int c = 0; c < 6; ++c)
            check(actual.virial[c], reference.virial[c], "strain derivative");
        }
      }
    }
    std::cout << "PASS " << (sizeof(rbmd::Real)==4 ? "float" : "double")
              << ": finite-difference forces/6 virials, energy/counts, periodic images, "
                 "owned/ghost permutations, empty partition, bounds guards\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
