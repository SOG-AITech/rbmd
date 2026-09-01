#include "mpi/mpi_distribution.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#ifdef READER_ENABLE_MPI

#include "mpi.h"

#include "parallel/include/atom_data_packet.h"
#include "common/rbmd_define.h"
#include "common/startup_phase_debug.h"

namespace reader::mpi {

namespace {

MPI_Comm Comm() { return MPI_COMM_WORLD; }

struct SpatialGrid {
  std::array<int, 3> dims{1, 1, 1};
  std::array<double, 3> coord_min{0.0, 0.0, 0.0};
  std::array<double, 3> lengths{1.0, 1.0, 1.0};
  int size = 1;
  MPI_Comm cart_comm = MPI_COMM_NULL;
  std::vector<int> coord_to_rank_map;

  SpatialGrid() = default;
  SpatialGrid(const SpatialGrid& other) { CopyFrom(other); }
  SpatialGrid& operator=(const SpatialGrid& other) {
    if (this != &other) {
      Reset();
      CopyFrom(other);
    }
    return *this;
  }
  SpatialGrid(SpatialGrid&& other) noexcept { MoveFrom(std::move(other)); }
  SpatialGrid& operator=(SpatialGrid&& other) noexcept {
    if (this != &other) {
      Reset();
      MoveFrom(std::move(other));
    }
    return *this;
  }
  ~SpatialGrid() { Reset(); }

 private:
  void Reset() {
    if (cart_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&cart_comm);
      cart_comm = MPI_COMM_NULL;
    }
  }

  void CopyFrom(const SpatialGrid& other) {
    dims = other.dims;
    coord_min = other.coord_min;
    lengths = other.lengths;
    size = other.size;
    coord_to_rank_map = other.coord_to_rank_map;
    cart_comm = MPI_COMM_NULL;
    if (other.cart_comm != MPI_COMM_NULL) {
      const int mpi_status = MPI_Comm_dup(other.cart_comm, &cart_comm);
      if (mpi_status != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Comm_dup failed");
      }
    }
  }

  void MoveFrom(SpatialGrid&& other) noexcept {
    dims = other.dims;
    coord_min = other.coord_min;
    lengths = other.lengths;
    size = other.size;
    cart_comm = other.cart_comm;
    coord_to_rank_map = std::move(other.coord_to_rank_map);
    other.cart_comm = MPI_COMM_NULL;
  }
};

bool StartupOwnerAuditEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_STARTUP_OWNER_AUDIT");
  if (!env) {
    return false;
  }
  if (Size() > 1) {
    return false;
  }
  std::string value(env);
  return !value.empty() && value != "0";
}

std::array<int, 3> NormalizeOwnerCoords(const SpatialGrid& grid,
                                        std::array<int, 3> coords) {
  for (int dim = 0; dim < 3; ++dim) {
    if (grid.dims[dim] <= 1) {
      coords[dim] = 0;
      continue;
    }
    coords[dim] = std::clamp(coords[dim], 0, grid.dims[dim] - 1);
  }
  return coords;
}

std::array<int, 3> ComputeAssignedOwnerCoords(
    const SpatialGrid& grid, const ParsedSnapshot::Atom& atom) {
  std::array<int, 3> coords{0, 0, 0};
  const double positions[3] = {static_cast<double>(atom.x),
                               static_cast<double>(atom.y),
                               static_cast<double>(atom.z)};
  for (int dim = 0; dim < 3; ++dim) {
    if (grid.dims[dim] <= 1) {
      continue;
    }
    const double relative =
        (positions[dim] - grid.coord_min[dim]) / grid.lengths[dim];
    const double scaled = relative * static_cast<double>(grid.dims[dim]);
    coords[dim] = static_cast<int>(std::floor(scaled));
  }
  return NormalizeOwnerCoords(grid, coords);
}

std::array<int, 3> ComputeExpectedOwnerCoords(
    const SpatialGrid& grid, const ParsedSnapshot::Atom& atom) {
  std::array<int, 3> coords{0, 0, 0};
  const double positions[3] = {static_cast<double>(atom.x),
                               static_cast<double>(atom.y),
                               static_cast<double>(atom.z)};
  for (int dim = 0; dim < 3; ++dim) {
    if (grid.dims[dim] <= 1) {
      continue;
    }
    coords[dim] = static_cast<int>(
        (positions[dim] - grid.coord_min[dim]) * grid.dims[dim] /
        grid.lengths[dim]);
  }
  return NormalizeOwnerCoords(grid, coords);
}

std::vector<int> BuildCartOwnerMap(const SpatialGrid& grid) {
  const int total_cells = grid.dims[0] * grid.dims[1] * grid.dims[2];
  std::vector<int> coord_to_rank_map(
      static_cast<std::size_t>(std::max(total_cells, 0)), 0);
  if (grid.size <= 1 || total_cells <= 0) {
    return coord_to_rank_map;
  }

  int dims[3] = {grid.dims[0], grid.dims[1], grid.dims[2]};
  int periods[3] = {1, 1, 1};
  int reorder = 0;
  MPI_Comm cart_comm = MPI_COMM_NULL;
  MPI_Cart_create(Comm(), 3, dims, periods, reorder, &cart_comm);
  if (cart_comm == MPI_COMM_NULL) {
    return coord_to_rank_map;
  }

  for (int z = 0; z < dims[2]; ++z) {
    for (int y = 0; y < dims[1]; ++y) {
      for (int x = 0; x < dims[0]; ++x) {
        int coords[3] = {x, y, z};
        int rank = 0;
        MPI_Cart_rank(cart_comm, coords, &rank);
        const int linear_index = x + dims[0] * (y + dims[1] * z);
        coord_to_rank_map[static_cast<std::size_t>(linear_index)] = rank;
      }
    }
  }

  MPI_Comm_free(&cart_comm);
  return coord_to_rank_map;
}

int ResolveOwnerFromCoords(const SpatialGrid& grid,
                           const std::vector<int>& coord_to_rank_map,
                           const std::array<int, 3>& coords) {
  long linear_index = coords[0];
  linear_index += static_cast<long>(grid.dims[0]) * coords[1];
  linear_index += static_cast<long>(grid.dims[0]) *
                  static_cast<long>(grid.dims[1]) * coords[2];
  if (linear_index < 0) {
    return 0;
  }
  const auto index = static_cast<std::size_t>(linear_index);
  if (index >= coord_to_rank_map.size()) {
    return std::max(grid.size - 1, 0);
  }
  return coord_to_rank_map[index];
}

SpatialGrid BuildSpatialGrid(const ParsedSnapshot::Box& box, int size) {
  SpatialGrid grid;
  grid.size = std::max(size, 1);
  if (grid.size > 1) {
    grid.dims = {0, 0, 0};
    MPI_Dims_create(grid.size, 3, grid.dims.data());
  }
  for (int dim = 0; dim < 3; ++dim) {
    if (grid.dims[dim] <= 0) {
      grid.dims[dim] = 1;
    }
  }
  grid.coord_min = {static_cast<double>(box.xlo), static_cast<double>(box.ylo),
                    static_cast<double>(box.zlo)};
  grid.lengths = {std::max(static_cast<double>(box.xhi - box.xlo),
                           std::numeric_limits<double>::epsilon()),
                  std::max(static_cast<double>(box.yhi - box.ylo),
                           std::numeric_limits<double>::epsilon()),
                  std::max(static_cast<double>(box.zhi - box.zlo),
                           std::numeric_limits<double>::epsilon())};

  if (grid.size > 1) {
    grid.coord_to_rank_map = BuildCartOwnerMap(grid);
  }
  return grid;
}

void EnsureDebugDirectoryExists() {
  if (mkdir("logs", 0777) != 0 && errno != EEXIST) {
    return;
  }
  (void)mkdir("logs/debug", 0777);
}

int DetermineOwner(const SpatialGrid& grid, const ParsedSnapshot::Atom& atom) {
  if (grid.size <= 1) {
    return 0;
  }

  std::array<int, 3> owner_coords{0, 0, 0};
  const double positions[3] = {static_cast<double>(atom.x),
                               static_cast<double>(atom.y),
                               static_cast<double>(atom.z)};
  for (int dim = 0; dim < 3; ++dim) {
    if (grid.dims[dim] <= 1) {
      continue;
    }

    owner_coords[dim] = static_cast<int>(
        (positions[dim] - grid.coord_min[dim]) * grid.dims[dim] /
        grid.lengths[dim]);
    if (owner_coords[dim] < 0) {
      owner_coords[dim] = 0;
    }
    if (owner_coords[dim] >= grid.dims[dim]) {
      owner_coords[dim] = grid.dims[dim] - 1;
    }
  }

  if (!grid.coord_to_rank_map.empty()) {
    return ResolveOwnerFromCoords(grid, grid.coord_to_rank_map, owner_coords);
  }

  return ResolveOwnerFromCoords(grid, BuildCartOwnerMap(grid), owner_coords);
}

void BroadcastBox(ParsedSnapshot& snapshot, int rank) {
  double values[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (rank == 0) {
    values[0] = snapshot.box.xlo;
    values[1] = snapshot.box.xhi;
    values[2] = snapshot.box.ylo;
    values[3] = snapshot.box.yhi;
    values[4] = snapshot.box.zlo;
    values[5] = snapshot.box.zhi;
  }
  MPI_Bcast(values, 6, MPI_DOUBLE, 0, Comm());
  if (rank != 0) {
    snapshot.box.xlo = values[0];
    snapshot.box.xhi = values[1];
    snapshot.box.ylo = values[2];
    snapshot.box.yhi = values[3];
    snapshot.box.zlo = values[4];
    snapshot.box.zhi = values[5];
  }
}

void BroadcastTopology(ParsedSnapshot& snapshot, int rank) {
  rbmd::Id values[5] = {0, 0, 0, 0, 0};
  if (rank == 0) {
    values[0] = snapshot.topology.atom_types;
    values[1] = snapshot.topology.bond_types;
    values[2] = snapshot.topology.angle_types;
    values[3] = snapshot.topology.dihedral_types;
    values[4] = snapshot.topology.improper_types;
  }
  MPI_Bcast(values, 5, MPI_RBMD_ID, 0, Comm());
  if (rank != 0) {
    snapshot.topology.atom_types = values[0];
    snapshot.topology.bond_types = values[1];
    snapshot.topology.angle_types = values[2];
    snapshot.topology.dihedral_types = values[3];
    snapshot.topology.improper_types = values[4];
  }
}

void BroadcastMasses(ParsedSnapshot& snapshot, int rank) {
  rbmd::Id mass_count = 0;
  if (rank == 0) {
    mass_count = static_cast<rbmd::Id>(snapshot.masses.size());
  }
  MPI_Bcast(&mass_count, 1, MPI_RBMD_ID, 0, Comm());

  const auto count = static_cast<int>(mass_count);
  std::vector<rbmd::Id> types(static_cast<std::size_t>(std::max(count, 0)));
  std::vector<rbmd::Real> masses(static_cast<std::size_t>(std::max(count, 0)));

  if (rank == 0) {
    std::size_t index = 0;
    for (const auto& entry : snapshot.masses) {
      types[index] = entry.first;
      masses[index] = entry.second;
      ++index;
    }
  }

  MPI_Bcast(count > 0 ? types.data() : nullptr, count, MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(count > 0 ? masses.data() : nullptr, count, MPI_RBMD_REAL, 0,
            Comm());

  if (rank != 0) {
    snapshot.masses.clear();
    for (int i = 0; i < count; ++i) {
      snapshot.masses.emplace(types[static_cast<std::size_t>(i)],
                              masses[static_cast<std::size_t>(i)]);
    }
  }
}

void BroadcastPairCoefficients(ParsedSnapshot& snapshot, int rank) {
  rbmd::Id count = 0;
  if (rank == 0) {
    count = static_cast<rbmd::Id>(snapshot.pair_coeffs.size());
  }
  MPI_Bcast(&count, 1, MPI_RBMD_ID, 0, Comm());

  std::vector<rbmd::Id> types(
      static_cast<std::size_t>(std::max<rbmd::Id>(count, 0)));
  std::vector<rbmd::Real> eps(types.size());
  std::vector<rbmd::Real> sigma(types.size());

  if (rank == 0) {
    std::size_t index = 0;
    for (const auto& entry : snapshot.pair_coeffs) {
      types[index] = entry.first;
      eps[index] = entry.second.epsilon;
      sigma[index] = entry.second.sigma;
      ++index;
    }
  }

  MPI_Bcast(count > 0 ? types.data() : nullptr, static_cast<int>(count),
            MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(count > 0 ? eps.data() : nullptr, static_cast<int>(count),
            MPI_RBMD_REAL, 0, Comm());
  MPI_Bcast(count > 0 ? sigma.data() : nullptr, static_cast<int>(count),
            MPI_RBMD_REAL, 0, Comm());

  if (rank != 0) {
    snapshot.pair_coeffs.clear();
    for (rbmd::Id i = 0; i < count; ++i) {
      snapshot.pair_coeffs[types[static_cast<std::size_t>(i)]] =
          ParsedSnapshot::PairCoeff{types[static_cast<std::size_t>(i)],
                                    eps[static_cast<std::size_t>(i)],
                                    sigma[static_cast<std::size_t>(i)]};
    }
  }
}

void BroadcastBondCoefficients(ParsedSnapshot& snapshot, int rank) {
  rbmd::Id count = 0;
  if (rank == 0) {
    count = static_cast<rbmd::Id>(snapshot.bond_coeffs.size());
  }
  MPI_Bcast(&count, 1, MPI_RBMD_ID, 0, Comm());

  std::vector<rbmd::Id> types(
      static_cast<std::size_t>(std::max<rbmd::Id>(count, 0)));
  std::vector<rbmd::Real> values(types.size() * 2);

  if (rank == 0) {
    std::size_t index = 0;
    for (const auto& entry : snapshot.bond_coeffs) {
      types[index] = entry.first;
      values[index * 2] = entry.second.k;
      values[index * 2 + 1] = entry.second.equilibrium;
      ++index;
    }
  }

  MPI_Bcast(count > 0 ? types.data() : nullptr, static_cast<int>(count),
            MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(count > 0 ? values.data() : nullptr, static_cast<int>(count * 2),
            MPI_RBMD_REAL, 0, Comm());

  if (rank != 0) {
    snapshot.bond_coeffs.clear();
    for (rbmd::Id i = 0; i < count; ++i) {
      snapshot.bond_coeffs[types[static_cast<std::size_t>(i)]] =
          ParsedSnapshot::BondCoeff{
              types[static_cast<std::size_t>(i)],
              values[static_cast<std::size_t>(i * 2)],
              values[static_cast<std::size_t>(i * 2 + 1)]};
    }
  }
}

void BroadcastAngleCoefficients(ParsedSnapshot& snapshot, int rank) {
  rbmd::Id count = 0;
  if (rank == 0) {
    count = static_cast<rbmd::Id>(snapshot.angle_coeffs.size());
  }
  MPI_Bcast(&count, 1, MPI_RBMD_ID, 0, Comm());

  std::vector<rbmd::Id> types(
      static_cast<std::size_t>(std::max<rbmd::Id>(count, 0)));
  std::vector<rbmd::Real> values(types.size() * 2);

  if (rank == 0) {
    std::size_t index = 0;
    for (const auto& entry : snapshot.angle_coeffs) {
      types[index] = entry.first;
      values[index * 2] = entry.second.k;
      values[index * 2 + 1] = entry.second.equilibrium;
      ++index;
    }
  }

  MPI_Bcast(count > 0 ? types.data() : nullptr, static_cast<int>(count),
            MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(count > 0 ? values.data() : nullptr, static_cast<int>(count * 2),
            MPI_RBMD_REAL, 0, Comm());

  if (rank != 0) {
    snapshot.angle_coeffs.clear();
    for (rbmd::Id i = 0; i < count; ++i) {
      snapshot.angle_coeffs[types[static_cast<std::size_t>(i)]] =
          ParsedSnapshot::AngleCoeff{
              types[static_cast<std::size_t>(i)],
              values[static_cast<std::size_t>(i * 2)],
              values[static_cast<std::size_t>(i * 2 + 1)]};
    }
  }
}

void BroadcastDihedralCoefficients(ParsedSnapshot& snapshot, int rank) {
  rbmd::Id count = 0;
  if (rank == 0) {
    count = static_cast<rbmd::Id>(snapshot.dihedral_coeffs.size());
  }
  MPI_Bcast(&count, 1, MPI_RBMD_ID, 0, Comm());

  std::vector<rbmd::Id> types(
      static_cast<std::size_t>(std::max<rbmd::Id>(count, 0)));
  std::vector<rbmd::Real> values(types.size() * 3);

  if (rank == 0) {
    std::size_t index = 0;
    for (const auto& entry : snapshot.dihedral_coeffs) {
      types[index] = entry.first;
      values[index * 3] = entry.second.k;
      values[index * 3 + 1] = entry.second.sign;
      values[index * 3 + 2] = entry.second.multiplicity;
      ++index;
    }
  }

  MPI_Bcast(count > 0 ? types.data() : nullptr, static_cast<int>(count),
            MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(count > 0 ? values.data() : nullptr, static_cast<int>(count * 3),
            MPI_RBMD_REAL, 0, Comm());

  if (rank != 0) {
    snapshot.dihedral_coeffs.clear();
    for (rbmd::Id i = 0; i < count; ++i) {
      snapshot.dihedral_coeffs[types[static_cast<std::size_t>(i)]] =
          ParsedSnapshot::DihedralCoeff{
              types[static_cast<std::size_t>(i)],
              values[static_cast<std::size_t>(i * 3)],
              values[static_cast<std::size_t>(i * 3 + 1)],
              values[static_cast<std::size_t>(i * 3 + 2)]};
    }
  }
}

void BroadcastImproperCoefficients(ParsedSnapshot& snapshot, int rank) {
  rbmd::Id count = 0;
  if (rank == 0) {
    count = static_cast<rbmd::Id>(snapshot.improper_coeffs.size());
  }
  MPI_Bcast(&count, 1, MPI_RBMD_ID, 0, Comm());

  const auto vector_size =
      static_cast<std::size_t>(std::max<rbmd::Id>(count, 0));
  std::vector<rbmd::Id> types(vector_size);
  std::vector<rbmd::Real> real_values(vector_size * 2);
  std::vector<rbmd::Id> integral_values(vector_size * 2);

  if (rank == 0) {
    std::size_t index = 0;
    for (const auto& entry : snapshot.improper_coeffs) {
      types[index] = entry.first;
      real_values[index * 2] = entry.second.k;
      real_values[index * 2 + 1] = entry.second.degree;
      integral_values[index * 2] = entry.second.d;
      integral_values[index * 2 + 1] = entry.second.n;
      ++index;
    }
  }

  MPI_Bcast(count > 0 ? types.data() : nullptr, static_cast<int>(count),
            MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(count > 0 ? real_values.data() : nullptr,
            static_cast<int>(count * 2), MPI_RBMD_REAL, 0, Comm());
  MPI_Bcast(count > 0 ? integral_values.data() : nullptr,
            static_cast<int>(count * 2), MPI_RBMD_ID, 0, Comm());

  if (rank != 0) {
    snapshot.improper_coeffs.clear();
    for (rbmd::Id i = 0; i < count; ++i) {
      ParsedSnapshot::ImproperCoeff coeff;
      const auto offset = static_cast<std::size_t>(i) * 2;
      coeff.type = types[static_cast<std::size_t>(i)];
      coeff.k = real_values[offset];
      coeff.degree = real_values[offset + 1];
      coeff.d = integral_values[offset];
      coeff.n = integral_values[offset + 1];
      snapshot.improper_coeffs[coeff.type] = coeff;
    }
  }
}


AtomPartition ScatterAtoms(ParsedSnapshot& snapshot, int rank, int size) {
  rbmd::debug::StartupPhaseLog("reader.mpi.scatter_atoms.begin",
                               "rank=" + std::to_string(rank) +
                                   " size=" + std::to_string(size));
  rbmd::Id total_atoms = 0;
  if (rank == 0) {
    total_atoms = static_cast<rbmd::Id>(snapshot.atoms.size());
  }
  MPI_Bcast(&total_atoms, 1, MPI_RBMD_ID, 0, Comm());

  AtomPartition partition;
  if (size <= 0) {
    snapshot.atoms.clear();
    return partition;
  }

  partition.offsets.resize(static_cast<std::size_t>(size) + 1);
  const SpatialGrid spatial = BuildSpatialGrid(snapshot.box, size);
  auto determine_owner = [&](const ParsedSnapshot::Atom& atom) -> int {
    return DetermineOwner(spatial, atom);
  };

  std::vector<int> send_counts;
  std::vector<int> displs;
  std::vector<FullAtomDataPacket> send_buffer;
  std::vector<std::pair<rbmd::Id, int>> assignments;

  if (size > 0) {
    if (rank == 0) {
      send_counts.assign(static_cast<std::size_t>(size), 0);
      displs.assign(static_cast<std::size_t>(size), 0);
      assignments.reserve(static_cast<std::size_t>(total_atoms));
      std::vector<std::vector<FullAtomDataPacket>> buckets(
          static_cast<std::size_t>(size));
      for (const auto& atom : snapshot.atoms) {
        const int owner = determine_owner(atom);
        assignments.emplace_back(atom.id, owner);

        FullAtomDataPacket packet{};
        packet.atom_id = atom.id;
        packet.atom_type = atom.type;
        packet.px = atom.x;
        packet.py = atom.y;
        packet.pz = atom.z;
        packet.vx = atom.vx;
        packet.vy = atom.vy;
        packet.vz = atom.vz;
        packet.charge = atom.charge;
        packet.molecules_id = atom.molecule;
        packet.image_x = atom.ix;
        packet.image_y = atom.iy;
        packet.image_z = atom.iz;
        buckets[static_cast<std::size_t>(owner)].push_back(packet);
      }

      rbmd::Id prefix = 0;
      for (int i = 0; i < size; ++i) {
        partition.offsets[static_cast<std::size_t>(i)] = prefix;
        const int count =
            static_cast<int>(buckets[static_cast<std::size_t>(i)].size());
        send_counts[static_cast<std::size_t>(i)] = count;
        prefix += static_cast<rbmd::Id>(count);
      }
      partition.offsets[static_cast<std::size_t>(size)] = prefix;

      send_buffer.resize(static_cast<std::size_t>(prefix));
      int displacement = 0;
      for (int i = 0; i < size; ++i) {
        displs[static_cast<std::size_t>(i)] = displacement;
        const auto& bucket = buckets[static_cast<std::size_t>(i)];
        std::copy(bucket.begin(), bucket.end(),
                  send_buffer.begin() + displacement);
        displacement += static_cast<int>(bucket.size());
      }

      std::sort(assignments.begin(), assignments.end(),
                [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
                });
      partition.atom_ids.resize(assignments.size());
      partition.owners.resize(assignments.size());
      for (std::size_t i = 0; i < assignments.size(); ++i) {
        partition.atom_ids[i] = assignments[i].first;
        partition.owners[i] = assignments[i].second;
      }
    } else {
      partition.offsets.resize(static_cast<std::size_t>(size) + 1);
    }
    if (!partition.offsets.empty()) {
      MPI_Bcast(partition.offsets.data(),
                static_cast<int>(partition.offsets.size()), MPI_RBMD_ID, 0,
                Comm());
    }
    rbmd::debug::StartupPhaseLog(
        "reader.mpi.scatter_atoms.after_offsets_bcast");
  }

  const auto local_begin =
      partition.offsets.empty()
          ? rbmd::Id{0}
          : partition.offsets[static_cast<std::size_t>(rank)];
  const auto local_end =
      partition.offsets.empty()
          ? total_atoms
          : partition.offsets[static_cast<std::size_t>(rank + 1)];
  const int local_count = static_cast<int>(local_end - local_begin);

  std::vector<FullAtomDataPacket> recv_buffer(
      static_cast<std::size_t>(std::max(local_count, 0)));

  MPI_Datatype mpi_atom_type;
  CreateMpiFullAtomDataPacket(&mpi_atom_type);

  MPI_Scatterv(rank == 0 ? send_buffer.data() : nullptr,
               rank == 0 ? send_counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, mpi_atom_type,
               local_count > 0 ? recv_buffer.data() : nullptr, local_count,
               mpi_atom_type, 0, Comm());
  rbmd::debug::StartupPhaseLog("reader.mpi.scatter_atoms.after_scatterv",
                               "local_count=" + std::to_string(local_count));

  MPI_Type_free(&mpi_atom_type);

  rbmd::Id assignment_count = 0;
  if (rank == 0) {
    assignment_count = static_cast<rbmd::Id>(partition.atom_ids.size());
  }
  MPI_Bcast(&assignment_count, 1, MPI_RBMD_ID, 0, Comm());
  partition.atom_ids.resize(static_cast<std::size_t>(assignment_count));
  partition.owners.resize(static_cast<std::size_t>(assignment_count));
  MPI_Bcast(partition.atom_ids.empty() ? nullptr : partition.atom_ids.data(),
            static_cast<int>(assignment_count), MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(partition.owners.empty() ? nullptr : partition.owners.data(),
            static_cast<int>(assignment_count), MPI_INT, 0, Comm());
  rbmd::debug::StartupPhaseLog(
      "reader.mpi.scatter_atoms.after_assignment_bcast");

  snapshot.atoms.clear();
  snapshot.atoms.reserve(static_cast<std::size_t>(std::max(local_count, 0)));
  for (int i = 0; i < local_count; ++i) {
    const auto& packet = recv_buffer[static_cast<std::size_t>(i)];
    ParsedSnapshot::Atom atom;
    atom.id = packet.atom_id;
    atom.type = packet.atom_type;
    atom.molecule = packet.molecules_id;
    atom.charge = packet.charge;
    atom.x = packet.px;
    atom.y = packet.py;
    atom.z = packet.pz;
    atom.vx = packet.vx;
    atom.vy = packet.vy;
    atom.vz = packet.vz;
    atom.ix = packet.image_x;
    atom.iy = packet.image_y;
    atom.iz = packet.image_z;
    snapshot.atoms.push_back(atom);
  }
  rbmd::debug::StartupPhaseLog("reader.mpi.scatter_atoms.end",
                               "local_atoms=" + std::to_string(local_count));

  return partition;
}

template <typename T, typename AppendFn, typename AssignFn>
void BroadcastStructured(std::vector<T>& container, int fields_per_entry,
                         int rank, AppendFn append_fn, AssignFn assign_fn) {
  rbmd::Id count = 0;
  if (rank == 0) {
    count = static_cast<rbmd::Id>(container.size());
  }
  MPI_Bcast(&count, 1, MPI_RBMD_ID, 0, Comm());

  std::vector<rbmd::Id> flat;
  if (rank == 0) {
    flat.reserve(static_cast<std::size_t>(count) * fields_per_entry);
    for (const auto& entry : container) {
      append_fn(flat, entry);
    }
  }
  flat.resize(static_cast<std::size_t>(count) * fields_per_entry);
  MPI_Bcast(flat.empty() ? nullptr : flat.data(), static_cast<int>(flat.size()),
            MPI_RBMD_ID, 0, Comm());

  container.clear();
  container.resize(static_cast<std::size_t>(count));
  for (std::size_t i = 0; i < container.size(); ++i) {
    assign_fn(container[i], flat, fields_per_entry, i);
  }
}

template <typename T, typename DetermineOwnerFn, typename AppendFn,
          typename AssignFn>
void ScatterStructured(std::vector<T>& container, int rank, int size,
                       DetermineOwnerFn determine_owner, int fields_per_entry,
                       AppendFn append_fn, AssignFn assign_fn) {
  if (size <= 0) {
    container.clear();
    return;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  std::vector<rbmd::Id> send_flat;

  if (rank == 0) {
    counts.assign(static_cast<std::size_t>(size), 0);
    displs.assign(static_cast<std::size_t>(size), 0);
    std::vector<std::vector<rbmd::Id>> buckets(static_cast<std::size_t>(size));
    for (const auto& entry : container) {
      int owner = determine_owner(entry);
      owner = std::clamp(owner, 0, std::max(size - 1, 0));
      append_fn(buckets[static_cast<std::size_t>(owner)], entry);
    }
    int displacement = 0;
    for (int i = 0; i < size; ++i) {
      const auto index = static_cast<std::size_t>(i);
      counts[index] = static_cast<int>(buckets[index].size());
      displs[index] = displacement;
      displacement += counts[index];
    }
    send_flat.reserve(static_cast<std::size_t>(std::max(displacement, 0)));
    for (int i = 0; i < size; ++i) {
      const auto& bucket = buckets[static_cast<std::size_t>(i)];
      send_flat.insert(send_flat.end(), bucket.begin(), bucket.end());
    }
  }

  int local_field_count = 0;
  MPI_Scatter(rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
              &local_field_count, 1, MPI_INT, 0, Comm());

  std::vector<rbmd::Id> recv_flat(
      static_cast<std::size_t>(std::max(local_field_count, 0)));
  MPI_Scatterv(rank == 0 ? send_flat.data() : nullptr,
               rank == 0 ? counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_RBMD_ID,
               local_field_count > 0 ? recv_flat.data() : nullptr,
               local_field_count, MPI_RBMD_ID, 0, Comm());

  const std::size_t entry_count =
      fields_per_entry > 0
          ? recv_flat.size() / static_cast<std::size_t>(fields_per_entry)
          : 0;
  container.clear();
  container.resize(entry_count);
  for (std::size_t i = 0; i < entry_count; ++i) {
    assign_fn(container[i], recv_flat, fields_per_entry, i);
  }
}

template <typename T, typename DetermineOwnersFn, typename AppendFn,
          typename AssignFn>
void ScatterStructuredMultiOwner(std::vector<T>& container, int rank, int size,
                                 DetermineOwnersFn determine_owners,
                                 int fields_per_entry, AppendFn append_fn,
                                 AssignFn assign_fn) {
  if (size <= 0) {
    container.clear();
    return;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  std::vector<rbmd::Id> send_flat;

  if (rank == 0) {
    counts.assign(static_cast<std::size_t>(size), 0);
    displs.assign(static_cast<std::size_t>(size), 0);
    std::vector<std::vector<rbmd::Id>> buckets(static_cast<std::size_t>(size));

    for (const auto& entry : container) {
      std::vector<int> owners = determine_owners(entry);
      std::sort(owners.begin(), owners.end());
      owners.erase(std::unique(owners.begin(), owners.end()), owners.end());

      for (int owner : owners) {
        owner = std::clamp(owner, 0, std::max(size - 1, 0));
        append_fn(buckets[static_cast<std::size_t>(owner)], entry);
      }
    }

    int displacement = 0;
    for (int i = 0; i < size; ++i) {
      const auto index = static_cast<std::size_t>(i);
      counts[index] = static_cast<int>(buckets[index].size());
      displs[index] = displacement;
      displacement += counts[index];
    }
    send_flat.reserve(static_cast<std::size_t>(std::max(displacement, 0)));
    for (int i = 0; i < size; ++i) {
      const auto& bucket = buckets[static_cast<std::size_t>(i)];
      send_flat.insert(send_flat.end(), bucket.begin(), bucket.end());
    }
  }

  int local_field_count = 0;
  MPI_Scatter(rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
              &local_field_count, 1, MPI_INT, 0, Comm());

  std::vector<rbmd::Id> recv_flat(
      static_cast<std::size_t>(std::max(local_field_count, 0)));
  MPI_Scatterv(rank == 0 ? send_flat.data() : nullptr,
               rank == 0 ? counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_RBMD_ID,
               local_field_count > 0 ? recv_flat.data() : nullptr,
               local_field_count, MPI_RBMD_ID, 0, Comm());

  const std::size_t entry_count =
      fields_per_entry > 0
          ? recv_flat.size() / static_cast<std::size_t>(fields_per_entry)
          : 0;
  container.clear();
  container.resize(entry_count);
  for (std::size_t i = 0; i < entry_count; ++i) {
    assign_fn(container[i], recv_flat, fields_per_entry, i);
  }
}

int OwnerRank(const AtomPartition& partition, rbmd::Id atom_id) {
  // 边界检查
  if (partition.atom_ids.empty() || partition.owners.empty()) {
    // 没有分区信息时，默认返回 rank 0
    throw std::runtime_error("not data");
  }

  if (atom_id < 0) {
    throw std::runtime_error("atoms id illegal");
  }

  auto it = std::lower_bound(partition.atom_ids.begin(),
                             partition.atom_ids.end(), atom_id);
  if (it != partition.atom_ids.end() && *it == atom_id) {
    // 精确匹配：返回对应的所有者
    const auto index =
        static_cast<std::size_t>(std::distance(partition.atom_ids.begin(), it));
    if (index < partition.owners.size()) {
      return partition.owners[index];
    }
  } else {
    throw std::runtime_error("can not find atoms id");
  }
  // 找不到原子 ID：这是错误情况，应该在调试版本中捕获
  // 生产环境中返回 rank 0 作为降级处理

  return 0;
}

struct VelocityPacket {
  rbmd::Id atom_id = 0;
  rbmd::Real vx = 0.0;
  rbmd::Real vy = 0.0;
  rbmd::Real vz = 0.0;
};

struct BondPacket {
  rbmd::Id id = 0;
  rbmd::Id type = 0;
  rbmd::Id atom1 = 0;
  rbmd::Id atom2 = 0;
};

struct AnglePacket {
  rbmd::Id id = 0;
  rbmd::Id type = 0;
  rbmd::Id atom1 = 0;
  rbmd::Id atom2 = 0;
  rbmd::Id atom3 = 0;
};

struct DihedralPacket {
  rbmd::Id id = 0;
  rbmd::Id type = 0;
  rbmd::Id atom1 = 0;
  rbmd::Id atom2 = 0;
  rbmd::Id atom3 = 0;
  rbmd::Id atom4 = 0;
};

struct ImproperPacket {
  rbmd::Id id = 0;
  rbmd::Id type = 0;
  rbmd::Id atom1 = 0;
  rbmd::Id atom2 = 0;
  rbmd::Id atom3 = 0;
  rbmd::Id atom4 = 0;
};

enum class StreamTag : int {
  Atom = 100,
  Velocity = 101,
  Bond = 102,
  Angle = 103,
  Dihedral = 104,
  Improper = 105
};

constexpr std::size_t kAtomBufferCapacity = 256;
constexpr std::size_t kOtherBufferCapacity = 256;

void ScatterBonds(ParsedSnapshot& snapshot, int rank,
                  const AtomPartition& partition) {
  const int size = partition.offsets.empty()
                       ? 0
                       : static_cast<int>(partition.offsets.size() - 1);
  ScatterStructuredMultiOwner<ParsedSnapshot::Bond>(
      snapshot.bonds, rank, size,
      [&](const ParsedSnapshot::Bond& bond) -> std::vector<int> {
        return {OwnerRank(partition, bond.atom1),
                OwnerRank(partition, bond.atom2)};
      },
      4,
      [](std::vector<rbmd::Id>& flat, const ParsedSnapshot::Bond& bond) {
        flat.push_back(bond.id);
        flat.push_back(bond.type);
        flat.push_back(bond.atom1);
        flat.push_back(bond.atom2);
      },
      [](ParsedSnapshot::Bond& bond, const std::vector<rbmd::Id>& flat,
         int stride, std::size_t index) {
        const std::size_t offset = index * static_cast<std::size_t>(stride);
        bond.id = flat[offset];
        bond.type = flat[offset + 1];
        bond.atom1 = flat[offset + 2];
        bond.atom2 = flat[offset + 3];
      });
}

void ScatterAngles(ParsedSnapshot& snapshot, int rank,
                   const AtomPartition& partition) {
  const int size = partition.offsets.empty()
                       ? 0
                       : static_cast<int>(partition.offsets.size() - 1);
  ScatterStructuredMultiOwner<ParsedSnapshot::Angle>(
      snapshot.angles, rank, size,
      [&](const ParsedSnapshot::Angle& angle) -> std::vector<int> {
        return {OwnerRank(partition, angle.atom1),
                OwnerRank(partition, angle.atom2),
                OwnerRank(partition, angle.atom3)};
      },
      5,
      [](std::vector<rbmd::Id>& flat, const ParsedSnapshot::Angle& angle) {
        flat.push_back(angle.id);
        flat.push_back(angle.type);
        flat.push_back(angle.atom1);
        flat.push_back(angle.atom2);
        flat.push_back(angle.atom3);
      },
      [](ParsedSnapshot::Angle& angle, const std::vector<rbmd::Id>& flat,
         int stride, std::size_t index) {
        const std::size_t offset = index * static_cast<std::size_t>(stride);
        angle.id = flat[offset];
        angle.type = flat[offset + 1];
        angle.atom1 = flat[offset + 2];
        angle.atom2 = flat[offset + 3];
        angle.atom3 = flat[offset + 4];
      });
}

void ScatterDihedrals(ParsedSnapshot& snapshot, int rank,
                      const AtomPartition& partition) {
  const int size = partition.offsets.empty()
                       ? 0
                       : static_cast<int>(partition.offsets.size() - 1);
  ScatterStructuredMultiOwner<ParsedSnapshot::Dihedral>(
      snapshot.dihedrals, rank, size,
      [&](const ParsedSnapshot::Dihedral& dihedral) -> std::vector<int> {
        return {OwnerRank(partition, dihedral.atom1),
                OwnerRank(partition, dihedral.atom2),
                OwnerRank(partition, dihedral.atom3),
                OwnerRank(partition, dihedral.atom4)};
      },
      6,
      [](std::vector<rbmd::Id>& flat,
         const ParsedSnapshot::Dihedral& dihedral) {
        flat.push_back(dihedral.id);
        flat.push_back(dihedral.type);
        flat.push_back(dihedral.atom1);
        flat.push_back(dihedral.atom2);
        flat.push_back(dihedral.atom3);
        flat.push_back(dihedral.atom4);
      },
      [](ParsedSnapshot::Dihedral& dihedral, const std::vector<rbmd::Id>& flat,
         int stride, std::size_t index) {
        const std::size_t offset = index * static_cast<std::size_t>(stride);
        dihedral.id = flat[offset];
        dihedral.type = flat[offset + 1];
        dihedral.atom1 = flat[offset + 2];
        dihedral.atom2 = flat[offset + 3];
        dihedral.atom3 = flat[offset + 4];
        dihedral.atom4 = flat[offset + 5];
      });
}

void ScatterImpropers(ParsedSnapshot& snapshot, int rank,
                      const AtomPartition& partition) {
  const int size = partition.offsets.empty()
                       ? 0
                       : static_cast<int>(partition.offsets.size() - 1);
  ScatterStructuredMultiOwner<ParsedSnapshot::Improper>(
      snapshot.impropers, rank, size,
      [&](const ParsedSnapshot::Improper& improper) -> std::vector<int> {
        return {OwnerRank(partition, improper.atom1),
                OwnerRank(partition, improper.atom2),
                OwnerRank(partition, improper.atom3),
                OwnerRank(partition, improper.atom4)};
      },
      6,
      [](std::vector<rbmd::Id>& flat,
         const ParsedSnapshot::Improper& improper) {
        flat.push_back(improper.id);
        flat.push_back(improper.type);
        flat.push_back(improper.atom1);
        flat.push_back(improper.atom2);
        flat.push_back(improper.atom3);
        flat.push_back(improper.atom4);
      },
      [](ParsedSnapshot::Improper& improper, const std::vector<rbmd::Id>& flat,
         int stride, std::size_t index) {
        const std::size_t offset = index * static_cast<std::size_t>(stride);
        improper.id = flat[offset];
        improper.type = flat[offset + 1];
        improper.atom1 = flat[offset + 2];
        improper.atom2 = flat[offset + 3];
        improper.atom3 = flat[offset + 4];
        improper.atom4 = flat[offset + 5];
      });
}

}  // namespace

struct StreamCtx::Impl {
  explicit Impl(int world_size)
      : atom_buffers(static_cast<std::size_t>(world_size)),
        velocity_buffers(static_cast<std::size_t>(world_size)),
        bond_buffers(static_cast<std::size_t>(world_size)),
        angle_buffers(static_cast<std::size_t>(world_size)),
        dihedral_buffers(static_cast<std::size_t>(world_size)),
        improper_buffers(static_cast<std::size_t>(world_size)),
        atom_counts(static_cast<std::size_t>(world_size), 0) {}

  std::vector<std::vector<FullAtomDataPacket>> atom_buffers;
  std::vector<std::vector<VelocityPacket>> velocity_buffers;
  std::vector<std::vector<BondPacket>> bond_buffers;
  std::vector<std::vector<AnglePacket>> angle_buffers;
  std::vector<std::vector<DihedralPacket>> dihedral_buffers;
  std::vector<std::vector<ImproperPacket>> improper_buffers;
  SpatialGrid spatial;
  std::vector<rbmd::Id> atom_counts;
  std::vector<std::pair<rbmd::Id, int>> assignments;
  std::unordered_map<rbmd::Id, int> owner_lookup;
  std::vector<int> owner_audit_cart_map;
  bool first_atom_logged = false;
  bool first_send_logged = false;
  bool atoms_closed = false;
  bool velocities_closed = false;
  bool bonds_closed = false;
  bool angles_closed = false;
  bool dihedrals_closed = false;
  bool impropers_closed = false;
  char dummy = 0;
};

namespace {

template <typename Packet>
void FlushPacketBuffer(StreamCtx& ctx, std::vector<Packet>& buffer,
                       int destination, StreamTag tag) {
  if (buffer.empty() || destination == ctx.rank) {
    return;
  }

  // 防止整数溢出
  const std::size_t total_bytes = buffer.size() * sizeof(Packet);
  if (total_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    //  应该分批发送或报错
    throw std::runtime_error("over max size");
    // // 当前简化处理：截断到最大允许大小
    // buffer.resize(std::numeric_limits<int>::max() / sizeof(Packet));
  }

  const auto bytes = static_cast<int>(buffer.size() * sizeof(Packet));
  if (bytes <= 0) {
    return;
  }
  if (ctx.impl && !ctx.impl->first_send_logged) {
    ctx.impl->first_send_logged = true;
    rbmd::debug::StartupPhaseLog(
        "reader.mpi.flush_packet_buffer.first_send",
        "destination=" + std::to_string(destination) + " tag=" +
            std::to_string(static_cast<int>(tag)) + " bytes=" +
            std::to_string(bytes));
  }

  MPI_Ssend(buffer.data(), bytes, MPI_BYTE, destination, static_cast<int>(tag),
            Comm());
  buffer.clear();
}

void SendSectionEnd(StreamCtx& ctx, int destination, StreamTag tag) {
  if (destination == ctx.rank) {
    return;
  }
  MPI_Ssend(&(ctx.impl->dummy), 0, MPI_BYTE, destination, static_cast<int>(tag),
            Comm());
}

void CloseAtoms(StreamCtx& ctx) {
  if (!ctx.impl || ctx.impl->atoms_closed) {
    return;
  }
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    FlushPacketBuffer(ctx,
                      ctx.impl->atom_buffers[static_cast<std::size_t>(dest)],
                      dest, StreamTag::Atom);
  }
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    SendSectionEnd(ctx, dest, StreamTag::Atom);
  }
  ctx.impl->atoms_closed = true;
}

void CloseVelocities(StreamCtx& ctx) {
  if (!ctx.impl || ctx.impl->velocities_closed) {
    return;
  }
  CloseAtoms(ctx);
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    FlushPacketBuffer(
        ctx, ctx.impl->velocity_buffers[static_cast<std::size_t>(dest)], dest,
        StreamTag::Velocity);
  }
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    SendSectionEnd(ctx, dest, StreamTag::Velocity);
  }
  ctx.impl->velocities_closed = true;
}

void CloseBonds(StreamCtx& ctx) {
  if (!ctx.impl || ctx.impl->bonds_closed) {
    return;
  }
  CloseVelocities(ctx);
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    FlushPacketBuffer(ctx,
                      ctx.impl->bond_buffers[static_cast<std::size_t>(dest)],
                      dest, StreamTag::Bond);
  }
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    SendSectionEnd(ctx, dest, StreamTag::Bond);
  }
  ctx.impl->bonds_closed = true;
}

void CloseAngles(StreamCtx& ctx) {
  if (!ctx.impl || ctx.impl->angles_closed) {
    return;
  }
  CloseBonds(ctx);
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    FlushPacketBuffer(ctx,
                      ctx.impl->angle_buffers[static_cast<std::size_t>(dest)],
                      dest, StreamTag::Angle);
  }
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    SendSectionEnd(ctx, dest, StreamTag::Angle);
  }
  ctx.impl->angles_closed = true;
}

void CloseDihedrals(StreamCtx& ctx) {
  if (!ctx.impl || ctx.impl->dihedrals_closed) {
    return;
  }
  CloseAngles(ctx);
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    FlushPacketBuffer(
        ctx, ctx.impl->dihedral_buffers[static_cast<std::size_t>(dest)], dest,
        StreamTag::Dihedral);
  }
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    SendSectionEnd(ctx, dest, StreamTag::Dihedral);
  }
  ctx.impl->dihedrals_closed = true;
}

void CloseImpropers(StreamCtx& ctx) {
  if (!ctx.impl || ctx.impl->impropers_closed) {
    return;
  }
  CloseDihedrals(ctx);
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    FlushPacketBuffer(
        ctx, ctx.impl->improper_buffers[static_cast<std::size_t>(dest)], dest,
        StreamTag::Improper);
  }
  for (int dest = 0; dest < ctx.size; ++dest) {
    if (dest == ctx.rank) {
      continue;
    }
    SendSectionEnd(ctx, dest, StreamTag::Improper);
  }
  ctx.impl->impropers_closed = true;
}

void EnsureAtomsClosed(StreamCtx& ctx) {
  if (ctx.impl && !ctx.impl->atoms_closed) {
    CloseAtoms(ctx);
  }
}

void EnsureVelocitiesClosed(StreamCtx& ctx) {
  if (ctx.impl && !ctx.impl->velocities_closed) {
    CloseVelocities(ctx);
  }
}

void EnsureBondsClosed(StreamCtx& ctx) {
  if (ctx.impl && !ctx.impl->bonds_closed) {
    CloseBonds(ctx);
  }
}

void EnsureAnglesClosed(StreamCtx& ctx) {
  if (ctx.impl && !ctx.impl->angles_closed) {
    CloseAngles(ctx);
  }
}

void EnsureDihedralsClosed(StreamCtx& ctx) {
  if (ctx.impl && !ctx.impl->dihedrals_closed) {
    CloseDihedrals(ctx);
  }
}

void EnsureImpropersClosed(StreamCtx& ctx) {
  if (ctx.impl && !ctx.impl->impropers_closed) {
    CloseImpropers(ctx);
  }
}

int ResolveOwner(const StreamCtx& ctx, rbmd::Id atom_id) {
  if (ctx.impl) {
    auto it = ctx.impl->owner_lookup.find(atom_id);
    if (it != ctx.impl->owner_lookup.end()) {
      return std::clamp(it->second, 0, std::max(ctx.size - 1, 0));
    }
  }
  return OwnerRank(ctx.partition, atom_id);
}

void AppendStartupOwnerAuditRow(StreamCtx& ctx,
                                const ParsedSnapshot::Atom& atom,
                                int assigned_owner) {
  if (!ctx.is_root || !ctx.impl || !StartupOwnerAuditEnabled()) {
    return;
  }

  const auto assigned_coords = ComputeAssignedOwnerCoords(ctx.impl->spatial, atom);
  const auto expected_coords = ComputeExpectedOwnerCoords(ctx.impl->spatial, atom);
  const int expected_owner =
      ResolveOwnerFromCoords(ctx.impl->spatial, ctx.impl->owner_audit_cart_map,
                             expected_coords);
  const int owner_match = assigned_owner == expected_owner ? 1 : 0;

  EnsureDebugDirectoryExists();
  static bool header_written = false;
  static std::string audit_path;
  if (audit_path.empty()) {
    audit_path = "logs/debug/startup_owner_audit_rank" +
                 std::to_string(ctx.rank) + "_pid" +
                 std::to_string(static_cast<long long>(getpid())) + ".csv";
  }

  std::ofstream out(audit_path, std::ios::app);
  if (!out) {
    return;
  }
  if (!header_written) {
    out << "atom_id,px,py,pz,assigned_owner,expected_owner,owner_match,"
           "assigned_coord_x,assigned_coord_y,assigned_coord_z,"
           "expected_coord_x,expected_coord_y,expected_coord_z\n";
    header_written = true;
  }
  out << atom.id << "," << atom.x << "," << atom.y << "," << atom.z << ","
      << assigned_owner << "," << expected_owner << "," << owner_match << ","
      << assigned_coords[0] << "," << assigned_coords[1] << ","
      << assigned_coords[2] << "," << expected_coords[0] << ","
      << expected_coords[1] << "," << expected_coords[2] << "\n";
}

}  // namespace

StreamCtx BeginStreaming(ParsedSnapshot& snapshot, rbmd::Id expected_atoms) {
  StreamCtx ctx;
  ctx.snapshot = &snapshot;

  if (!IsEnabled()) {
    return ctx;
  }

  ctx.rank = Rank();
  ctx.size = Size();
  ctx.is_root = (ctx.rank == 0);
  ctx.active = true;
  rbmd::debug::StartupPhaseLog("reader.mpi.begin_streaming.begin",
                               "expected_atoms=" +
                                   std::to_string(static_cast<long long>(
                                       expected_atoms)));

  // 在流式传输开始前广播 box 信息，确保所有进程都有空间划分依据
  BroadcastBox(snapshot, ctx.rank);
  rbmd::debug::StartupPhaseLog(
      "reader.mpi.begin_streaming.after_broadcast_box");
  BroadcastTopology(snapshot, ctx.rank);
  rbmd::debug::StartupPhaseLog(
      "reader.mpi.begin_streaming.after_broadcast_topology");
  rbmd::Id total_atoms = 0;
  if (ctx.is_root) {
    total_atoms = expected_atoms;
  }
  MPI_Bcast(&total_atoms, 1, MPI_RBMD_ID, 0, Comm());
  rbmd::debug::StartupPhaseLog(
      "reader.mpi.begin_streaming.after_broadcast_total_atoms");
  rbmd::debug::StartupPhaseLog(
      "reader.mpi.begin_streaming.before_build_spatial_grid",
      "size=" + std::to_string(ctx.size));
  SpatialGrid streaming_spatial = BuildSpatialGrid(snapshot.box, ctx.size);
  rbmd::debug::StartupPhaseLog(
      "reader.mpi.begin_streaming.after_build_spatial_grid",
      "dims=" + std::to_string(streaming_spatial.dims[0]) + "x" +
          std::to_string(streaming_spatial.dims[1]) + "x" +
          std::to_string(streaming_spatial.dims[2]));
  const auto safe_total =
      total_atoms > 0 ? total_atoms : static_cast<rbmd::Id>(0);
  const bool owner_audit_enabled = StartupOwnerAuditEnabled();
  std::vector<int> startup_owner_audit_cart_map;
  if (owner_audit_enabled) {
    startup_owner_audit_cart_map = streaming_spatial.coord_to_rank_map;
    if (startup_owner_audit_cart_map.empty()) {
      startup_owner_audit_cart_map = BuildCartOwnerMap(streaming_spatial);
    }
  }
  ctx.partition.offsets.assign(static_cast<std::size_t>(ctx.size) + 1, 0);
  ctx.partition.atom_ids.clear();
  ctx.partition.owners.clear();

  snapshot.atoms.clear();
  snapshot.bonds.clear();
  snapshot.angles.clear();
  snapshot.dihedrals.clear();
  snapshot.impropers.clear();

  if (ctx.size > 0) {
    const rbmd::Id base =
        safe_total / static_cast<rbmd::Id>(std::max(ctx.size, 1));
    const rbmd::Id extra =
        ctx.rank < (safe_total % static_cast<rbmd::Id>(std::max(ctx.size, 1)))
            ? 1
            : 0;
    // 增加 20-50% 的安全余量
    constexpr double kSafetyFactor = 1.3;  // 可配置
    const auto estimated = static_cast<std::size_t>(
        static_cast<double>(base + extra) * kSafetyFactor);

    if (estimated > 0) {
      snapshot.atoms.reserve(std::min(estimated,
          static_cast<std::size_t>(safe_total)));  // 不超过总数
    }
  }

  ctx.local_index.clear();

  if (ctx.is_root) {
    ctx.impl = std::make_shared<StreamCtx::Impl>(ctx.size);
    ctx.impl->spatial = std::move(streaming_spatial);
    ctx.impl->owner_audit_cart_map = std::move(startup_owner_audit_cart_map);
    ctx.impl->atom_counts.assign(static_cast<std::size_t>(ctx.size), 0);
    ctx.impl->assignments.clear();
    ctx.impl->owner_lookup.clear();
    const auto expected =
        static_cast<std::size_t>(std::max<rbmd::Id>(safe_total, 0));
    ctx.impl->assignments.reserve(expected);
    ctx.impl->owner_lookup.reserve(expected);
  }
  rbmd::debug::StartupPhaseLog("reader.mpi.begin_streaming.end",
                               "is_root=" + std::to_string(ctx.is_root ? 1 : 0));
  return ctx;
}

void StreamAtom(StreamCtx& ctx, const ParsedSnapshot::Atom& atom) {
  if (!ctx.snapshot) {
    return;
  }
  if (!IsEnabled() || !ctx.active) {
    ctx.snapshot->atoms.push_back(atom);
    ctx.local_index[atom.id] = ctx.snapshot->atoms.size() - 1;
    return;
  }

  if (!ctx.is_root) {
    return;
  }

  if (!ctx.impl) {
    ctx.snapshot->atoms.push_back(atom);
    ctx.local_index[atom.id] = ctx.snapshot->atoms.size() - 1;
    return;
  }

  const int owner_raw = DetermineOwner(ctx.impl->spatial, atom);
  const int owner = std::clamp(owner_raw, 0, std::max(ctx.size - 1, 0));
  if (!ctx.impl->first_atom_logged) {
    ctx.impl->first_atom_logged = true;
    rbmd::debug::StartupPhaseLog(
        "reader.mpi.stream_atom.first_atom",
        "id=" + std::to_string(static_cast<long long>(atom.id)) +
            " owner=" + std::to_string(owner));
  }
  ctx.impl->assignments.emplace_back(atom.id, owner);
  ctx.impl->owner_lookup[atom.id] = owner;
  AppendStartupOwnerAuditRow(ctx, atom, owner);
  if (!ctx.impl->atom_counts.empty()) {
    ctx.impl->atom_counts[static_cast<std::size_t>(owner)] += 1;
  }

  if (owner == ctx.rank) {
    ctx.snapshot->atoms.push_back(atom);
    ctx.local_index[atom.id] = ctx.snapshot->atoms.size() - 1;
    return;
  }

  auto& buffer = ctx.impl->atom_buffers[static_cast<std::size_t>(owner)];
  FullAtomDataPacket packet{};
  packet.atom_id = atom.id;
  packet.atom_type = atom.type;
  packet.px = atom.x;
  packet.py = atom.y;
  packet.pz = atom.z;
  packet.vx = atom.vx;
  packet.vy = atom.vy;
  packet.vz = atom.vz;
  packet.charge = atom.charge;
  packet.molecules_id = atom.molecule;
  packet.image_x = atom.ix;
  packet.image_y = atom.iy;
  packet.image_z = atom.iz;
  buffer.push_back(packet);
  if (buffer.size() >= kAtomBufferCapacity) {
    FlushPacketBuffer(ctx, buffer, owner, StreamTag::Atom);
  }
}

void StreamVelocity(StreamCtx& ctx, rbmd::Id atom_id, rbmd::Real vx,
                    rbmd::Real vy, rbmd::Real vz) {
  if (!ctx.snapshot) {
    return;
  }
  if (!IsEnabled() || !ctx.active) {
    auto it = ctx.local_index.find(atom_id);
    if (it != ctx.local_index.end()) {
      auto& atom = ctx.snapshot->atoms[it->second];
      atom.vx = vx;
      atom.vy = vy;
      atom.vz = vz;
    }
    return;
  }

  if (!ctx.is_root) {
    return;
  }

  EnsureAtomsClosed(ctx);

  const int owner = ResolveOwner(ctx, atom_id);
  if (owner == ctx.rank || !ctx.impl) {
    auto it = ctx.local_index.find(atom_id);
    if (it != ctx.local_index.end()) {
      auto& atom = ctx.snapshot->atoms[it->second];
      atom.vx = vx;
      atom.vy = vy;
      atom.vz = vz;
    }
    return;
  }

  auto& buffer = ctx.impl->velocity_buffers[static_cast<std::size_t>(owner)];
  VelocityPacket packet{};
  packet.atom_id = atom_id;
  packet.vx = vx;
  packet.vy = vy;
  packet.vz = vz;
  buffer.push_back(packet);
  if (buffer.size() >= kOtherBufferCapacity) {
    FlushPacketBuffer(ctx, buffer, owner, StreamTag::Velocity);
  }
}

void StreamBond(StreamCtx& ctx, const ParsedSnapshot::Bond& bond) {
  if (!ctx.snapshot) {
    return;
  }
  if (!IsEnabled() || !ctx.active) {
    ctx.snapshot->bonds.push_back(bond);
    return;
  }

  if (!ctx.is_root) {
    return;
  }

  EnsureVelocitiesClosed(ctx);

  const int owner1 = ResolveOwner(ctx, bond.atom1);
  const int owner2 = ResolveOwner(ctx, bond.atom2);

  if (owner1 == ctx.rank || !ctx.impl) {
    ctx.snapshot->bonds.push_back(bond);
  } else {
    auto& buffer = ctx.impl->bond_buffers[static_cast<std::size_t>(owner1)];
    BondPacket packet{};
    packet.id = bond.id;
    packet.type = bond.type;
    packet.atom1 = bond.atom1;
    packet.atom2 = bond.atom2;
    buffer.push_back(packet);
    if (buffer.size() >= kOtherBufferCapacity) {
      FlushPacketBuffer(ctx, buffer, owner1, StreamTag::Bond);
    }
  }

  if (owner2 != owner1) {
    if (owner2 == ctx.rank || !ctx.impl) {
      ctx.snapshot->bonds.push_back(bond);
    } else {
      auto& buffer = ctx.impl->bond_buffers[static_cast<std::size_t>(owner2)];
      BondPacket packet{};
      packet.id = bond.id;
      packet.type = bond.type;
      packet.atom1 = bond.atom1;
      packet.atom2 = bond.atom2;
      buffer.push_back(packet);
      if (buffer.size() >= kOtherBufferCapacity) {
        FlushPacketBuffer(ctx, buffer, owner2, StreamTag::Bond);
      }
    }
  }
}

void StreamAngle(StreamCtx& ctx, const ParsedSnapshot::Angle& angle) {
  if (!ctx.snapshot) {
    return;
  }
  if (!IsEnabled() || !ctx.active) {
    ctx.snapshot->angles.push_back(angle);
    return;
  }

  if (!ctx.is_root) {
    return;
  }

  EnsureBondsClosed(ctx);

  const int owner1 = ResolveOwner(ctx, angle.atom1);
  const int owner2 = ResolveOwner(ctx, angle.atom2);
  const int owner3 = ResolveOwner(ctx, angle.atom3);

  std::vector<int> owners = {owner1, owner2, owner3};
  std::sort(owners.begin(), owners.end());
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());

  for (int owner : owners) {
    if (owner == ctx.rank || !ctx.impl) {
      ctx.snapshot->angles.push_back(angle);
    } else {
      auto& buffer = ctx.impl->angle_buffers[static_cast<std::size_t>(owner)];
      AnglePacket packet{};
      packet.id = angle.id;
      packet.type = angle.type;
      packet.atom1 = angle.atom1;
      packet.atom2 = angle.atom2;
      packet.atom3 = angle.atom3;
      buffer.push_back(packet);
      if (buffer.size() >= kOtherBufferCapacity) {
        FlushPacketBuffer(ctx, buffer, owner, StreamTag::Angle);
      }
    }
  }
}

void StreamDihedral(StreamCtx& ctx, const ParsedSnapshot::Dihedral& dihedral) {
  if (!ctx.snapshot) {
    return;
  }
  if (!IsEnabled() || !ctx.active) {
    ctx.snapshot->dihedrals.push_back(dihedral);
    return;
  }

  if (!ctx.is_root) {
    return;
  }

  EnsureAnglesClosed(ctx);

  const int owner1 = ResolveOwner(ctx, dihedral.atom1);
  const int owner2 = ResolveOwner(ctx, dihedral.atom2);
  const int owner3 = ResolveOwner(ctx, dihedral.atom3);
  const int owner4 = ResolveOwner(ctx, dihedral.atom4);

  std::vector<int> owners = {owner1, owner2, owner3, owner4};
  std::sort(owners.begin(), owners.end());
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());

  for (int owner : owners) {
    if (owner == ctx.rank || !ctx.impl) {
      ctx.snapshot->dihedrals.push_back(dihedral);
    } else {
      auto& buffer =
          ctx.impl->dihedral_buffers[static_cast<std::size_t>(owner)];
      DihedralPacket packet{};
      packet.id = dihedral.id;
      packet.type = dihedral.type;
      packet.atom1 = dihedral.atom1;
      packet.atom2 = dihedral.atom2;
      packet.atom3 = dihedral.atom3;
      packet.atom4 = dihedral.atom4;
      buffer.push_back(packet);
      if (buffer.size() >= kOtherBufferCapacity) {
        FlushPacketBuffer(ctx, buffer, owner, StreamTag::Dihedral);
      }
    }
  }
}

void StreamImproper(StreamCtx& ctx, const ParsedSnapshot::Improper& improper) {
  if (!ctx.snapshot) {
    return;
  }
  if (!IsEnabled() || !ctx.active) {
    ctx.snapshot->impropers.push_back(improper);
    return;
  }

  if (!ctx.is_root) {
    return;
  }

  EnsureDihedralsClosed(ctx);

  const int owner1 = ResolveOwner(ctx, improper.atom1);
  const int owner2 = ResolveOwner(ctx, improper.atom2);
  const int owner3 = ResolveOwner(ctx, improper.atom3);
  const int owner4 = ResolveOwner(ctx, improper.atom4);

  std::vector<int> owners = {owner1, owner2, owner3, owner4};
  std::sort(owners.begin(), owners.end());
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());

  for (int owner : owners) {
    if (owner == ctx.rank || !ctx.impl) {
      ctx.snapshot->impropers.push_back(improper);
    } else {
      auto& buffer =
          ctx.impl->improper_buffers[static_cast<std::size_t>(owner)];
      ImproperPacket packet{};
      packet.id = improper.id;
      packet.type = improper.type;
      packet.atom1 = improper.atom1;
      packet.atom2 = improper.atom2;
      packet.atom3 = improper.atom3;
      packet.atom4 = improper.atom4;
      buffer.push_back(packet);
      if (buffer.size() >= kOtherBufferCapacity) {
        FlushPacketBuffer(ctx, buffer, owner, StreamTag::Improper);
      }
    }
  }
}

void ReceiveLoop(StreamCtx& ctx) {
  if (!IsEnabled() || ctx.rank == 0 || !ctx.snapshot) {
    return;
  }

  ctx.active = true;
  rbmd::debug::StartupPhaseLog("reader.mpi.receive_loop.begin");

  bool atoms_done = false;
  bool velocities_done = false;
  bool bonds_done = false;
  bool angles_done = false;
  bool dihedrals_done = false;
  bool impropers_done = false;

  bool first_probe_logged = false;
  while (!(atoms_done && velocities_done && bonds_done &&
          angles_done && dihedrals_done && impropers_done)) {
    MPI_Status status;
    // 直接阻塞等待消息
    MPI_Probe(0, MPI_ANY_TAG, Comm(), &status);
    if (!first_probe_logged) {
      first_probe_logged = true;
      rbmd::debug::StartupPhaseLog(
          "reader.mpi.receive_loop.after_first_probe",
          "tag=" + std::to_string(status.MPI_TAG));
    }
    int byte_count = 0;
    MPI_Get_count(&status, MPI_BYTE, &byte_count);

    const auto tag = static_cast<StreamTag>(status.MPI_TAG);
    if (byte_count == 0) {
      char dummy = 0;
      MPI_Recv(&dummy, 0, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, Comm(),
               MPI_STATUS_IGNORE);
      switch (tag) {
        case StreamTag::Atom:
          atoms_done = true;
          break;
        case StreamTag::Velocity:
          velocities_done = true;
          break;
        case StreamTag::Bond:
          bonds_done = true;
          break;
        case StreamTag::Angle:
          angles_done = true;
          break;
        case StreamTag::Dihedral:
          dihedrals_done = true;
          break;
        case StreamTag::Improper:
          impropers_done = true;
          break;
      }
      continue;
    }

    std::vector<char> payload(static_cast<std::size_t>(byte_count));
    MPI_Recv(payload.data(), byte_count, MPI_BYTE, status.MPI_SOURCE,
             status.MPI_TAG, Comm(), MPI_STATUS_IGNORE);

    switch (tag) {
      case StreamTag::Atom: {
        const auto entries =
            static_cast<std::size_t>(byte_count) / sizeof(FullAtomDataPacket);
        const auto* packets =
            reinterpret_cast<const FullAtomDataPacket*>(payload.data());
        for (std::size_t i = 0; i < entries; ++i) {
          ParsedSnapshot::Atom atom;
          atom.id = packets[i].atom_id;
          atom.type = packets[i].atom_type;
          atom.molecule = packets[i].molecules_id;
          atom.charge = packets[i].charge;
          atom.x = packets[i].px;
          atom.y = packets[i].py;
          atom.z = packets[i].pz;
          atom.vx = packets[i].vx;
          atom.vy = packets[i].vy;
          atom.vz = packets[i].vz;
          atom.ix = packets[i].image_x;
          atom.iy = packets[i].image_y;
          atom.iz = packets[i].image_z;
          ctx.local_index[atom.id] = ctx.snapshot->atoms.size();
          ctx.snapshot->atoms.push_back(atom);
        }
        break;
      }
      case StreamTag::Velocity: {
        const auto entries =
            static_cast<std::size_t>(byte_count) / sizeof(VelocityPacket);
        const auto* packets =
            reinterpret_cast<const VelocityPacket*>(payload.data());
        for (std::size_t i = 0; i < entries; ++i) {
          const auto id = packets[i].atom_id;
          auto it = ctx.local_index.find(id);
          if (it != ctx.local_index.end()) {
            auto& atom = ctx.snapshot->atoms[it->second];
            atom.vx = packets[i].vx;
            atom.vy = packets[i].vy;
            atom.vz = packets[i].vz;
          }
        }
        break;
      }
      case StreamTag::Bond: {
        const auto entries =
            static_cast<std::size_t>(byte_count) / sizeof(BondPacket);
        const auto* packets =
            reinterpret_cast<const BondPacket*>(payload.data());
        for (std::size_t i = 0; i < entries; ++i) {
          ParsedSnapshot::Bond bond;
          bond.id = packets[i].id;
          bond.type = packets[i].type;
          bond.atom1 = packets[i].atom1;
          bond.atom2 = packets[i].atom2;
          ctx.snapshot->bonds.push_back(bond);
        }
        break;
      }
      case StreamTag::Angle: {
        const auto entries =
            static_cast<std::size_t>(byte_count) / sizeof(AnglePacket);
        const auto* packets =
            reinterpret_cast<const AnglePacket*>(payload.data());
        for (std::size_t i = 0; i < entries; ++i) {
          ParsedSnapshot::Angle angle;
          angle.id = packets[i].id;
          angle.type = packets[i].type;
          angle.atom1 = packets[i].atom1;
          angle.atom2 = packets[i].atom2;
          angle.atom3 = packets[i].atom3;
          ctx.snapshot->angles.push_back(angle);
        }
        break;
      }
      case StreamTag::Dihedral: {
        const auto entries =
            static_cast<std::size_t>(byte_count) / sizeof(DihedralPacket);
        const auto* packets =
            reinterpret_cast<const DihedralPacket*>(payload.data());
        for (std::size_t i = 0; i < entries; ++i) {
          ParsedSnapshot::Dihedral dihedral;
          dihedral.id = packets[i].id;
          dihedral.type = packets[i].type;
          dihedral.atom1 = packets[i].atom1;
          dihedral.atom2 = packets[i].atom2;
          dihedral.atom3 = packets[i].atom3;
          dihedral.atom4 = packets[i].atom4;
          ctx.snapshot->dihedrals.push_back(dihedral);
        }
        break;
      }
      case StreamTag::Improper: {
        const auto entries =
            static_cast<std::size_t>(byte_count) / sizeof(ImproperPacket);
        const auto* packets =
            reinterpret_cast<const ImproperPacket*>(payload.data());
        for (std::size_t i = 0; i < entries; ++i) {
          ParsedSnapshot::Improper improper;
          improper.id = packets[i].id;
          improper.type = packets[i].type;
          improper.atom1 = packets[i].atom1;
          improper.atom2 = packets[i].atom2;
          improper.atom3 = packets[i].atom3;
          improper.atom4 = packets[i].atom4;
          ctx.snapshot->impropers.push_back(improper);
        }
        break;
      }
    }
  }

  ctx.active = false;
}

void EndStreaming(StreamCtx& ctx) {
  if (!IsEnabled() || !ctx.snapshot) {
    return;
  }
  rbmd::debug::StartupPhaseLog("reader.mpi.end_streaming.begin",
                               "is_root=" + std::to_string(ctx.is_root ? 1 : 0));

  if (ctx.is_root && ctx.impl) {
    EnsureImpropersClosed(ctx);
    ctx.partition.offsets.resize(static_cast<std::size_t>(ctx.size) + 1);
    rbmd::Id prefix = 0;
    if (!ctx.partition.offsets.empty()) {
      ctx.partition.offsets[0] = 0;
      for (int i = 0; i < ctx.size; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const rbmd::Id count = index < ctx.impl->atom_counts.size()
                                   ? ctx.impl->atom_counts[index]
                                   : static_cast<rbmd::Id>(0);
        prefix += count;
        ctx.partition.offsets[index + 1] = prefix;
      }
    }
    auto& assignments = ctx.impl->assignments;
    std::sort(
        assignments.begin(), assignments.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    // 验证排序结果（调试模式）
    assert(std::is_sorted(assignments.begin(), assignments.end(),
                          [](const auto& lhs, const auto& rhs) {
                            return lhs.first < rhs.first;
                          }));

    ctx.partition.atom_ids.resize(assignments.size());
    ctx.partition.owners.resize(assignments.size());
    for (std::size_t i = 0; i < assignments.size(); ++i) {
      ctx.partition.atom_ids[i] = assignments[i].first;
      ctx.partition.owners[i] = assignments[i].second;
    }
  }

  ctx.partition.offsets.resize(static_cast<std::size_t>(ctx.size) + 1);
  if (!ctx.partition.offsets.empty()) {
    MPI_Bcast(ctx.partition.offsets.data(),
              static_cast<int>(ctx.partition.offsets.size()), MPI_RBMD_ID, 0,
              Comm());
  }

  rbmd::Id assignment_count =
      ctx.is_root ? static_cast<rbmd::Id>(ctx.partition.atom_ids.size())
                  : static_cast<rbmd::Id>(0);
  MPI_Bcast(&assignment_count, 1, MPI_RBMD_ID, 0, Comm());
  ctx.partition.atom_ids.resize(static_cast<std::size_t>(assignment_count));
  ctx.partition.owners.resize(static_cast<std::size_t>(assignment_count));
  MPI_Bcast(
      ctx.partition.atom_ids.empty() ? nullptr : ctx.partition.atom_ids.data(),
      static_cast<int>(assignment_count), MPI_RBMD_ID, 0, Comm());
  MPI_Bcast(
      ctx.partition.owners.empty() ? nullptr : ctx.partition.owners.data(),
      static_cast<int>(assignment_count), MPI_INT, 0, Comm());

  const int rank = Rank();
  BroadcastMasses(*ctx.snapshot, rank);
  BroadcastPairCoefficients(*ctx.snapshot, rank);
  BroadcastBondCoefficients(*ctx.snapshot, rank);
  BroadcastAngleCoefficients(*ctx.snapshot, rank);
  BroadcastDihedralCoefficients(*ctx.snapshot, rank);
  BroadcastImproperCoefficients(*ctx.snapshot, rank);

  ctx.active = false;
}

bool IsEnabled() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return false;
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized != 0) {
    return false;
  }
  int size = 1;
  MPI_Comm_size(Comm(), &size);
  return size > 1;
}

int Rank() {
  if (!IsEnabled()) {
    return 0;
  }
  int rank = 0;
  MPI_Comm_rank(Comm(), &rank);
  return rank;
}

int Size() {
  if (!IsEnabled()) {
    return 1;
  }
  int size = 1;
  MPI_Comm_size(Comm(), &size);
  return size;
}

void DistributeSnapshot(ParsedSnapshot& snapshot) {
  if (!IsEnabled()) {
    return;
  }

  const int rank = Rank();
  const int size = Size();

  BroadcastBox(snapshot, rank);
  BroadcastTopology(snapshot, rank);
  BroadcastMasses(snapshot, rank);
  BroadcastPairCoefficients(snapshot, rank);
  BroadcastBondCoefficients(snapshot, rank);
  BroadcastAngleCoefficients(snapshot, rank);
  BroadcastDihedralCoefficients(snapshot, rank);
  BroadcastImproperCoefficients(snapshot, rank);
  AtomPartition partition = ScatterAtoms(snapshot, rank, size);
  ScatterBonds(snapshot, rank, partition);
  ScatterAngles(snapshot, rank, partition);
  ScatterDihedrals(snapshot, rank, partition);
  ScatterImpropers(snapshot, rank, partition);
}

}  // namespace reader::mpi

#else

namespace reader::mpi {

bool IsEnabled() { return false; }

int Rank() { return 0; }

int Size() { return 1; }

void DistributeSnapshot(ParsedSnapshot&) {}

StreamCtx BeginStreaming(ParsedSnapshot& snapshot, rbmd::Id) {
  StreamCtx ctx;
  ctx.snapshot = &snapshot;
  ctx.rank = 0;
  ctx.size = 1;
  ctx.is_root = true;
  ctx.active = false;
  return ctx;
}

void StreamAtom(StreamCtx& ctx, const ParsedSnapshot::Atom& atom) {
  if (!ctx.snapshot) {
    return;
  }
  ctx.snapshot->atoms.push_back(atom);
  ctx.local_index[atom.id] = ctx.snapshot->atoms.size() - 1;
}

void StreamVelocity(StreamCtx& ctx, rbmd::Id atom_id, rbmd::Real vx,
                    rbmd::Real vy, rbmd::Real vz) {
  if (!ctx.snapshot) {
    return;
  }
  auto it = ctx.local_index.find(atom_id);
  if (it != ctx.local_index.end()) {
    auto& atom = ctx.snapshot->atoms[it->second];
    atom.vx = vx;
    atom.vy = vy;
    atom.vz = vz;
  }
}

void StreamBond(StreamCtx& ctx, const ParsedSnapshot::Bond& bond) {
  if (ctx.snapshot) {
    ctx.snapshot->bonds.push_back(bond);
  }
}

void StreamAngle(StreamCtx& ctx, const ParsedSnapshot::Angle& angle) {
  if (ctx.snapshot) {
    ctx.snapshot->angles.push_back(angle);
  }
}

void StreamDihedral(StreamCtx& ctx, const ParsedSnapshot::Dihedral& dihedral) {
  if (ctx.snapshot) {
    ctx.snapshot->dihedrals.push_back(dihedral);
  }
}

void StreamImproper(StreamCtx& ctx, const ParsedSnapshot::Improper& improper) {
  if (ctx.snapshot) {
    ctx.snapshot->impropers.push_back(improper);
  }
}

void ReceiveLoop(StreamCtx&) {}

void EndStreaming(StreamCtx&) {}

}  // namespace reader::mpi

#endif
