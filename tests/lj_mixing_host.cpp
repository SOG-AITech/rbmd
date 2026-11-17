#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "common/types.h"
#include "common/lj_mixing.h"

namespace device { struct DEVICE_GPU {}; }
using STREAM_T = void*;
using R = rbmd::Real;
struct Box {};
struct Dim { unsigned int x = 0; } blockIdx, threadIdx, blockDim{1};
#define __global__
#define __device__
#define __shared__
#define BLOCK_SIZE 1
#define SQRT std::sqrt
#define POW std::pow
#define CHECK_KERNEL(...) __VA_ARGS__
template <typename T, int N> struct BLOCKREDUCE {
  struct TempStorage {};
  explicit BLOCKREDUCE(TempStorage&) {}
  T Sum(T value) { return value; }
};
template <typename T> void atomicAdd(T* target, T value) { *target += value; }
void MinImageDistance_while(Box, R&, R&, R&) {}
// Zero-charge fixtures isolate LJ. These shims do not test Coulomb arithmetic.
void CoulCutForce_fix(R, R, R, R qi, R qj, R, R, R,
                      R& a, R& b, R& c, R& d) {
  if (qi || qj) throw std::runtime_error("fixture must have zero charges");
  a = b = c = d = 0;
}
void CoulCutForceUser(R, R, R, R, R, R qi, R qj,
                      R, R, R, R, R, R, R, R, rbmd::Id, R,
                      R& a, R& b, R& c, R& d) {
  if (qi || qj) throw std::runtime_error("fixture must have zero charges");
  a = b = c = d = 0;
}
#include "production_lj.h"

void check(double actual, double expected) {
  const double tol = sizeof(R) == 4 ? 2e-5 : 2e-7;
  if (!std::isfinite(actual) || std::abs(actual - expected) > tol * (1 + std::abs(expected)))
    throw std::runtime_error("actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}
// Independent scalar potential, with deliberately dissimilar type parameters.
double potential(double r, bool geometric, double weight, bool same_type) {
  const double sigma = same_type ? 2.0 : (geometric ? 4.0 : 5.0);
  const double epsilon = same_type ? 0.25 : 0.5;
  const double s6 = std::pow(sigma / r, 6);
  return weight * 4 * epsilon * (s6*s6 - s6);
}
void exercise(bool geometric, int relation, bool same_type, R r) {
  rbmd::Id types[]{0, same_type ? 0 : 1}, ids[]{7, 23};
  rbmd::Id start[]{0, 1}, end[]{1, 2}, neighbors[]{1, 0}, special[]{23, 7};
  int counts[6]{};
  for (int atom = 0; atom < 2; ++atom)
    for (int k = 0; k < 3; ++k) counts[3*atom+k] = relation >= 0 && k >= relation;
  R sigma[]{2, 8}, eps[]{R(.25), 1}, x[]{0, r}, zero[2]{}, taylor[6]{};
  R f12[4]{}, f6[4]{};
  build_table(sigma, eps, geometric, f12, f6);
  const double weight = relation < 0 ? 1 : (relation == 2 ? .5 : 0);
  const double expected = r >= 10 ? 0 : potential(r, geometric, weight, same_type);
  const double h = 1e-4;
  const double derivative = r >= 10 ? 0 :
      (potential(r+h, geometric, weight, same_type) - potential(r-h, geometric, weight, same_type))/(2*h);
  R table_force, table_energy;
  const int index = same_type ? 0 : 1;
  LJForceEnergyFromTypePair(r*r, R(100), f12[index], f6[index], table_force, table_energy);
  check(2*weight*table_energy, expected);
  check(weight*table_force*r, derivative);
  check(f12[1], f12[2]); check(f6[1], f6[2]);
  for (bool sog : {false, true}) {
    R energy=0, coul=0, virial[12]{}, fx[2]{}, fy[2]{}, fz[2]{};
    for (blockIdx.x = 0; blockIdx.x < 2; ++blockIdx.x) {
      if (sog)
        ComputeSpecialLJCutCoulForceUserOp<device::DEVICE_GPU>()(
            Box{}, 10, 2, 1, 1, 2, 0, 0, taylor, types, ids, sigma, eps, geometric,
            start, end, neighbors, counts, special, 1, 0, 0, R(.5),
            zero, x, zero, zero, fx, fy, fz, virial, &energy, &coul);
      else
        SpeciaLJCutCoulEnergyOp<device::DEVICE_GPU>()(
            Box{}, 10, 2, 1, 1, types, ids, sigma, eps, geometric,
            start, end, neighbors, counts, special, 1, 0, 0, R(.5),
            zero, x, zero, zero, virial, &energy, &coul, nullptr);
    }
    check(energy, expected); check(coul, 0);
    check(virial[0]+virial[1], -r*derivative);
    if (sog) { check(fx[0], derivative); check(fx[1], -derivative); }
  }
}
int main() {
  check(rbmd::UseGeometricLJMixing("OPLS", ""), true);
  check(rbmd::UseGeometricLJMixing("HARMONIC", ""), false);
  check(rbmd::UseGeometricLJMixing("", "GEOMETRIC"), true);
  check(rbmd::UseGeometricLJMixing("OPLS", "ARITHMETIC"), false);
  bool rejected = false;
  try { rbmd::UseGeometricLJMixing("OPLS", "TYPO"); }
  catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, true);
  for (bool geometric : {false, true})
    for (bool same_type : {false, true})
      for (int relation : {-1, 0, 1, 2})
        for (R r : {R(4.7), R(6), R(11)}) exercise(geometric, relation, same_type, r);
  std::cout << "PASS LJ mixing/table/RBL-energy/SOG, special weights, forces/virials ("
            << sizeof(R)*8 << " bit)\n";
}
