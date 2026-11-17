#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "rbl_full_neighbor_list_op.h"

namespace {
using Id = rbmd::Id;
using Real = rbmd::Real;

void CheckHip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    if (size_ != 0) {
      CheckHip(hipMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)),
               "hipMalloc");
    }
  }
  explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
    if (!host.empty()) {
      CheckHip(hipMemcpy(data_, host.data(), host.size() * sizeof(T),
                         hipMemcpyHostToDevice),
               "hipMemcpy H2D");
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { if (data_) hipFree(data_); }
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::vector<T> ToHost() const {
    std::vector<T> host(size_);
    if (size_ != 0) {
      CheckHip(hipMemcpy(host.data(), data_, size_ * sizeof(T),
                         hipMemcpyDeviceToHost),
               "hipMemcpy D2H");
    }
    return host;
  }
  void Zero() {
    if (size_ != 0) CheckHip(hipMemset(data_, 0, size_ * sizeof(T)), "hipMemset");
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

std::uint64_t Mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

bool LegacySelected(Id frequency, Id epoch, Id atom_id, Id neighbor_id) {
  if (frequency <= 1) return true;
  std::uint64_t key = static_cast<std::uint64_t>(epoch);
  key ^= Mix(static_cast<std::uint64_t>(atom_id));
  key ^= Mix(static_cast<std::uint64_t>(neighbor_id) +
             0xd1b54a32d192ed03ULL);
  return Mix(key) % static_cast<std::uint64_t>(frequency) == 0;
}

struct Fixture {
  static constexpr Id kShellCount = 220;
  static constexpr Id kCore = 221;
  static constexpr Id kOutside = 222;
  static constexpr Id kExtra = 223;
  std::vector<Id> ids;
  std::vector<Real> x, y, z;
  std::vector<Id> full_candidates;
  std::vector<Id> empty_candidates;
  Box box;

  Fixture() : ids(224), x(224, Real(20)), y(224, Real(0)), z(224, Real(0)) {
    for (Id i = 0; i < static_cast<Id>(ids.size()); ++i) ids[i] = 1000 + i;
    x[0] = Real(0);
    for (Id i = 1; i <= kShellCount; ++i) {
      x[i] = Real(6.51) + Real(0.01) * (i % 220);
      full_candidates.push_back(i);
    }
    x[kCore] = Real(1);
    x[kOutside] = Real(10);
    full_candidates.insert(full_candidates.begin(), kCore);
    full_candidates.push_back(kOutside);
    full_candidates.push_back(0);
    empty_candidates = {kCore, kOutside, 0};
    box._pbc_x = box._pbc_y = box._pbc_z = false;
    box._length[0] = box._length[1] = box._length[2] = Real(100);
  }
};

std::vector<Id> Expected(const Fixture& f, const std::vector<Id>& candidates,
                         Id frequency, Id epoch) {
  std::vector<Id> result;
  for (Id neighbor : candidates) {
    if (neighbor == 0) continue;
    const Real dx = f.x[neighbor] - f.x[0];
    const Real r2 = dx * dx;
    if (r2 >= Real(6.5 * 6.5) - EPSILON &&
        r2 < Real(9.0 * 9.0) - EPSILON &&
        LegacySelected(frequency, epoch, f.ids[0], f.ids[neighbor])) {
      result.push_back(neighbor);
    }
  }
  return result;
}

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void RunCachedCase(bool implicit, Id frequency, bool empty_shell) {
  Fixture f;
  const Id epoch = 1;
  const auto& row = empty_shell ? f.empty_candidates : f.full_candidates;
  const auto expected = Expected(f, row, frequency, epoch);
  std::vector<Id> storage = row;
  Id start_value = 0;
  Id end_value = static_cast<Id>(row.size());
  if (!implicit) {
    storage.insert(storage.begin(), {15, 15});
    start_value = 2;
    end_value += 2;
  }
  DeviceBuffer<Id> d_ids(f.ids), d_start(std::vector<Id>{start_value}),
      d_end(std::vector<Id>{end_value}),
      d_num(std::vector<Id>{static_cast<Id>(row.size())}), d_candidates(storage);
  DeviceBuffer<Real> d_x(f.x), d_y(f.y), d_z(f.z);
  op::SampleRblNeighborCandidatesOp<device::DEVICE_GPU> sample;

  auto launch = [&](Id capacity, std::vector<Id>* values, Id* required,
                    Id* stored_num, Real* weight) {
    DeviceBuffer<Id> d_random(static_cast<std::size_t>(capacity));
    DeviceBuffer<Id> d_random_num(1), d_required(1);
    DeviceBuffer<Real> d_weight(1);
    d_random_num.Zero(); d_required.Zero(); d_weight.Zero();
    sample(Real(6.5 * 6.5) - EPSILON, Real(9.0 * 9.0) - EPSILON, 1,
           d_ids.data(), d_x.data(), d_y.data(), d_z.data(), d_start.data(),
           d_end.data(), d_candidates.data(), d_num.data(),
           static_cast<Id>(row.size()), implicit, capacity, d_random.data(),
           d_random_num.data(), d_weight.data(), d_required.data(), f.box,
           frequency, epoch, 8);
    CheckHip(hipDeviceSynchronize(), "cached kernel");
    *values = d_random.ToHost();
    *required = d_required.ToHost()[0];
    *stored_num = d_random_num.ToHost()[0];
    *weight = d_weight.ToHost()[0];
  };

  std::vector<Id> values;
  Id required = 0, stored = 0;
  Real weight = 0;
  constexpr Id initial_capacity = 100;
  launch(initial_capacity, &values, &required, &stored, &weight);
  Require(required == static_cast<Id>(expected.size()), "cached required mismatch");
  Require(stored == std::min<Id>(initial_capacity, required),
          "cached truncation mismatch");
  Require(weight == static_cast<Real>(frequency), "cached weight mismatch");
  const Id grown = std::max<Id>(initial_capacity, required);
  launch(grown, &values, &required, &stored, &weight);
  values.resize(static_cast<std::size_t>(stored));
  Require(values == expected, "cached expanded sample mismatch");
  Require(stored == required, "cached expanded count mismatch");
}

void RunDirectCase(Id threads_per_particle, Id frequency, bool empty_shell) {
  Fixture f;
  const Id epoch = 1;
  const auto& row = empty_shell ? f.empty_candidates : f.full_candidates;
  std::vector<Id> sorted{0};
  sorted.insert(sorted.end(), row.begin(), row.end());
  const auto expected = Expected(f, row, frequency, epoch);
  DeviceBuffer<Id> d_ids(f.ids), d_cell(std::vector<Id>{0}),
      d_cell_start(std::vector<Id>{0}),
      d_cell_end(std::vector<Id>{static_cast<Id>(sorted.size())}),
      d_sorted(sorted), d_neighbor_cell(std::vector<Id>{0}),
      d_neighbor_num(1), d_core_start(std::vector<Id>{0}), d_core_end(1),
      d_core(4), d_should_realloc(1), d_required_core(1);
  DeviceBuffer<Real> d_x(f.x), d_y(f.y), d_z(f.z);
  op::GenerateRblFullNeighborListOp<device::DEVICE_GPU> generate;

  auto launch = [&](Id capacity, std::vector<Id>* values, Id* required,
                    Id* stored_num, Real* weight) {
    DeviceBuffer<Id> d_random(static_cast<std::size_t>(capacity));
    DeviceBuffer<Id> d_random_num(1), d_required(1);
    DeviceBuffer<Real> d_weight(1);
    d_neighbor_num.Zero(); d_core_end.Zero(); d_should_realloc.Zero();
    d_required_core.Zero(); d_random_num.Zero(); d_required.Zero(); d_weight.Zero();
    generate(d_cell.data(), d_cell_start.data(), d_cell_end.data(),
             Real(6.5 * 6.5) - EPSILON, Real(9.0 * 9.0) - EPSILON, 1,
             d_ids.data(), d_x.data(), d_y.data(), d_z.data(), d_sorted.data(),
             d_neighbor_cell.data(), 1, d_neighbor_num.data(), 4,
             d_core_start.data(), d_core_end.data(), d_core.data(), capacity,
             d_random.data(), d_random_num.data(), d_weight.data(),
             d_required.data(), f.box, d_should_realloc.data(),
             d_required_core.data(), frequency, epoch, threads_per_particle);
    CheckHip(hipDeviceSynchronize(), "direct kernel");
    *values = d_random.ToHost();
    *required = d_required.ToHost()[0];
    *stored_num = d_random_num.ToHost()[0];
    *weight = d_weight.ToHost()[0];
    Require(d_should_realloc.ToHost()[0] == 0, "unexpected core reallocation");
    Require(d_core_end.ToHost()[0] == 1, "direct core list mismatch");
    Require(d_core.ToHost()[0] == Fixture::kCore,
            "direct core neighbor mismatch");
  };

  std::vector<Id> values;
  Id required = 0, stored = 0;
  Real weight = 0;
  constexpr Id initial_capacity = 100;
  launch(initial_capacity, &values, &required, &stored, &weight);
  Require(required == static_cast<Id>(expected.size()), "direct required mismatch");
  Require(stored == std::min<Id>(initial_capacity, required),
          "direct truncation mismatch");
  Require(weight == static_cast<Real>(frequency), "direct weight mismatch");
  const Id grown = std::max<Id>(initial_capacity, required);
  launch(grown, &values, &required, &stored, &weight);
  values.resize(static_cast<std::size_t>(stored));
  Require(values == expected, "direct expanded sample mismatch");
  Require(stored == required, "direct expanded count mismatch");
}
}  // namespace

int main() {
  try {
    RunCachedCase(true, 1, false);
    RunCachedCase(true, 2, false);
    RunCachedCase(false, 1, false);
    RunCachedCase(false, 2, false);
    RunCachedCase(true, 2, true);
    RunCachedCase(false, 2, true);
    RunDirectCase(4, 1, false);
    RunDirectCase(4, 2, false);
    RunDirectCase(8, 1, false);
    RunDirectCase(8, 2, false);
    RunDirectCase(4, 2, true);
    RunDirectCase(8, 2, true);
    std::cout << "RBL sampling kernel tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RBL sampling kernel tests: FAIL: " << error.what() << '\n';
    return 1;
  }
}
