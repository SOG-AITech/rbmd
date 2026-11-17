#include "SelectedTrajectoryOutput.h"

#include <array>
#include <chrono>
#include <experimental/filesystem>
#include <numeric>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include "Logger.hpp"

namespace {
constexpr size_t kChunkAtoms = 65536;
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template<class T>
void AppendSelected(const thrust::device_vector<T>& source,
                    const thrust::device_vector<size_t>& indices, size_t count,
                    thrust::device_vector<T>& scratch, std::vector<T>& target) {
  if (!count) return;
  thrust::gather(thrust::device, indices.begin(), indices.begin() + count,
                 source.begin(), scratch.begin());
  const size_t offset = target.size();
  target.resize(offset + count);
  // Only the compacted device buffer crosses the device/host boundary.
  thrust::copy(scratch.begin(), scratch.begin() + count, target.begin() + offset);
}
}  // namespace

void SelectedTrajectoryOutput::CollectiveCheck(bool ok, const std::string& message) const {
  int all_ok = ok ? 1 : 0;
#ifdef USE_MPI
  if (_world > 1) {
    int local_ok = all_ok;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
#endif
  if (!all_ok) throw std::runtime_error(message);
}

rbmd::GlobalCount SelectedTrajectoryOutput::GlobalSum(rbmd::GlobalCount local) const {
#ifdef USE_MPI
  if (_world > 1) {
    rbmd::GlobalCount global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return global;
  }
#endif
  return local;
}

void SelectedTrajectoryOutput::Init() {
#ifdef USE_MPI
  int initialized = 0, finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized) MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_world);
  }
#endif
  auto config = DataManager::getInstance().getConfigData();
  const auto& node = config->GetJsonNode("outputs").at("trajectory_out");
  _options = rbmd::trajectory::ParseOptions(node);
  _interval = node.at("interval").get<rbmd::Id>();
  if (_interval <= 0) throw std::runtime_error("selected trajectory interval must be positive");
  const bool sharded = node.value("mpi_sharded", _world > 1);
  if (_world > 1 && !sharded)
    throw std::runtime_error("selected trajectory requires mpi_sharded=true with multiple ranks");

  const std::string path = sharded ? "rbmd.rank" + std::to_string(_rank) + ".trj" : "rbmd.trj";
  namespace fs = std::experimental::filesystem;
  bool opened = false;
  try {
    const auto status = fs::symlink_status(path);
    if (!fs::exists(status) || fs::is_regular_file(status)) {
      _file.open(path, std::ios::out | std::ios::trunc);
      opened = _file.is_open();
    }
  } catch (const std::exception&) {}
  CollectiveCheck(opened, "Could not open selected trajectory file on all ranks");
  _ids = _options.ids;
  _types = _options.types;
  _regions = _options.regions;
  // Fixed-size scratch independent of the number of simulated atoms.
  _indices.resize(kChunkAtoms);
  _id_scratch.resize(kChunkAtoms);
  _real_scratch.resize(kChunkAtoms);
  Logger::Instance().info(
      "SelectedTrajectoryOutput: mode={} interval={} max_atoms={} ranks={} chunk_atoms={} fields={}",
      _options.initial ? "initial" : "dynamic", _interval, _options.max_atoms,
      _world, kChunkAtoms, _options.fields.size());
}

rbmd::trajectory::Predicate SelectedTrajectoryOutput::MakePredicate() {
  rbmd::trajectory::Predicate p;
  p.atom_ids = raw_ptr(_device_data->_d_atoms_id);
  p.atom_types = raw_ptr(_device_data->_d_atoms_type);
  p.x = raw_ptr(_device_data->_d_px);
  p.y = raw_ptr(_device_data->_d_py);
  p.z = raw_ptr(_device_data->_d_pz);
  p.ids = raw_ptr(_ids);
  p.num_ids = _ids.size();
  p.filter_ids = _frozen || _options.filter_ids;
  if (!_frozen) {
    p.types = raw_ptr(_types);
    p.num_types = _types.size();
    p.regions = raw_ptr(_regions);
    p.num_regions = _regions.size();
  }
  return p;
}

void SelectedTrajectoryOutput::FreezeIds(const std::vector<rbmd::Id>& local_ids) {
  std::vector<rbmd::Id> ids = local_ids;
#ifdef USE_MPI
  if (_world > 1) {
    // max_atoms <= INT_MAX was validated before counting/copying.
    const int count = static_cast<int>(local_ids.size());
    std::vector<int> counts(_world), displacements(_world);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total = 0;
    for (int r = 0; r < _world; ++r) {
      displacements[r] = total;
      total += counts[r];
    }
    ids.resize(total);
    MPI_Allgatherv(local_ids.data(), count, MPI_RBMD_ID, ids.data(), counts.data(),
                   displacements.data(), MPI_RBMD_ID, MPI_COMM_WORLD);
  }
#endif
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    throw std::runtime_error("duplicate native atom IDs in initial trajectory selection");
  _ids = ids;
  _frozen = true;
  Logger::Instance().info("SelectedTrajectoryOutput froze {} global atom IDs", ids.size());
}

void SelectedTrajectoryOutput::Execute(rbmd::Id timestep) {
  if (timestep % _interval != 0) return;
  // Finish simulation work before measuring output costs, and before reading
  // arrays that may have been updated on non-default streams.
  if (!CHECK_RUNTIME(DEVICESYNC()))
    throw std::runtime_error("device synchronization failed before trajectory selection");
  const auto start = Clock::now();
  const size_t native = static_cast<size_t>(*_structure_info_data->_num_atoms);
  const auto predicate = MakePredicate();
  size_t selected = 0;
  for (size_t offset = 0; offset < native; offset += kChunkAtoms) {
    const auto begin = thrust::make_counting_iterator<size_t>(offset);
    const size_t count = std::min(kChunkAtoms, native - offset);
    selected += thrust::count_if(thrust::device, begin, begin + count, predicate);
    // Once a rank alone exceeds the GLOBAL limit, no more scanning is needed.
    if (selected > _options.max_atoms) break;
  }
  const auto global_selected = GlobalSum(static_cast<rbmd::GlobalCount>(selected));
  if (global_selected > static_cast<rbmd::GlobalCount>(_options.max_atoms))
    throw std::runtime_error("trajectory selection exceeds global max_atoms=" +
        std::to_string(_options.max_atoms) + "; narrow the window or raise the explicit limit");
  const double scan_s = Seconds(start);
  const auto pack_start = Clock::now();

  std::vector<rbmd::Id> ids, types;
  std::array<std::vector<rbmd::Real>, 10> values;
  const std::array<const thrust::device_vector<rbmd::Real>*, 10> sources{
      &_device_data->_d_charge, &_device_data->_d_px, &_device_data->_d_py,
      &_device_data->_d_pz, &_device_data->_d_unwarp_px, &_device_data->_d_unwarp_py,
      &_device_data->_d_unwarp_pz, &_device_data->_d_vx, &_device_data->_d_vy,
      &_device_data->_d_vz};
  std::vector<size_t> columns;
  for (const auto& field : _options.fields) {
    const auto& all = rbmd::trajectory::AllFields();
    columns.push_back(static_cast<size_t>(std::find(all.begin(), all.end(), field) - all.begin()));
  }
  const bool need_types = std::find(columns.begin(), columns.end(), 1) != columns.end();
  ids.reserve(selected);  // IDs also order rows and support fixed-cohort tracking.
  if (need_types) types.reserve(selected);
  for (auto c : columns) if (c >= 2) values[c - 2].reserve(selected);
  if (selected) for (size_t offset = 0; offset < native; offset += kChunkAtoms) {
    const auto begin = thrust::make_counting_iterator<size_t>(offset);
    const size_t count = std::min(kChunkAtoms, native - offset);
    const auto end = thrust::copy_if(thrust::device, begin, begin + count, _indices.begin(), predicate);
    const size_t n = static_cast<size_t>(end - _indices.begin());
    AppendSelected(_device_data->_d_atoms_id, _indices, n, _id_scratch, ids);
    if (need_types) AppendSelected(_device_data->_d_atoms_type, _indices, n, _id_scratch, types);
    for (auto c : columns)
      if (c >= 2) AppendSelected(*sources[c - 2], _indices, n, _real_scratch, values[c - 2]);
  }
  if (_options.initial && !_frozen) FreezeIds(ids);
  const double pack_s = Seconds(pack_start);
  const auto write_start = Clock::now();

  const auto& box = *DataManager::getInstance().getMDData()->_box;
  rbmd::Real lo[3], hi[3];
  std::copy(box._coord_min, box._coord_min + 3, lo);
  std::copy(box._coord_max, box._coord_max + 3, hi);
#ifdef USE_MPI
  if (_world > 1) {
    MPI_Allreduce(MPI_IN_PLACE, lo, 3, MPI_RBMD_REAL, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, hi, 3, MPI_RBMD_REAL, MPI_MAX, MPI_COMM_WORLD);
  }
#endif
  const auto file_start = _file.tellp();
  _file << fmt::format("ITEM: TIMESTEP\n{}\nITEM: NUMBER OF ATOMS\n{}\n", timestep, ids.size());
  _file << fmt::format("ITEM: BOX BOUNDS {} {} {}\n", box._pbc_x ? "pp" : "ff",
                      box._pbc_y ? "pp" : "ff", box._pbc_z ? "pp" : "ff");
  for (int d = 0; d < 3; ++d) _file << fmt::format("{} {}\n", lo[d], hi[d]);
  _file << "ITEM: ATOMS";
  for (const auto& field : _options.fields) _file << ' ' << field;
  _file << '\n';
  std::vector<size_t> order(ids.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ids[a] < ids[b]; });
  std::string batch;
  batch.reserve(256 * 1024);
  for (auto i : order) {
    for (size_t j = 0; j < columns.size(); ++j) {
      if (j) batch += ' ';
      const auto c = columns[j];
      if (c == 0) batch += fmt::format("{}", ids[i] + 1);
      else if (c == 1) batch += fmt::format("{}", types[i] + 1);
      else batch += fmt::format("{}", values[c - 2][i]);
    }
    batch += '\n';
    if (batch.size() >= 128 * 1024) { _file << batch; batch.clear(); }
  }
  _file << batch;
  // Make every completed frame visible to readers; this is not an fsync.
  _file.flush();
  CollectiveCheck(_file.good(), "selected trajectory write failed on one or more ranks");
  const double write_s = Seconds(write_start);
  const auto bytes = static_cast<rbmd::GlobalCount>(_file.tellp() - file_start);
  double timing[4] = {scan_s, pack_s, write_s, Seconds(start)};
#ifdef USE_MPI
  if (_world > 1) MPI_Allreduce(MPI_IN_PLACE, timing, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  const auto global_bytes = GlobalSum(bytes);
  Logger::Instance().info(
      "TRAJECTORY_FRAME step={} atoms={} bytes={} scan_s={:.6f} pack_copy_s={:.6f} write_s={:.6f} total_s={:.6f}",
      timestep, global_selected, global_bytes, timing[0], timing[1], timing[2], timing[3]);
}
