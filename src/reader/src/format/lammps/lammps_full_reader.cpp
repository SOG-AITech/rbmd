#include "formats/lammps/lammps_full_reader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <experimental/filesystem>
#include <iterator>
#include <limits>
#include <random>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "data_manager.h"
#include "common/unit_factor.h"
#include "common/startup_phase_debug.h"
#include "common/mpi_root_guard.hpp"
#include "force_field/cvff_force_field_data.h"
#include "force_field/lj_force_field_data.h"
#include "formats/structure_adapters.h"
#include "io/icursor.h"
#include "model/box.h"
#include "model/md_data.h"
#include "model/structure_info_data.h"
#include "mpi/mpi_distribution.h"
#ifdef READER_ENABLE_MPI
#include "mpi/special_bond_builder.h"
#include "mpi/topology_converter.h"
#include "mpi.h"
#endif
#include "structure_data/atoms_structure_data.h"
#include "structure_data/basic_structure_data.h"
#include "structure_data/charge_structure_data.h"
#include "structure_data/full_structure_data.h"

#ifdef READER_ENABLE_MPI
#include "rbmd_parallel_until_locator.h"
#endif


namespace reader {

namespace {

using reader::detail::FillZero;
namespace fs = std::experimental::filesystem;

struct PairHash {
  std::size_t operator()(const std::pair<rbmd::Id, rbmd::Id>& value) const noexcept {
    // A simple hash combine for rbmd::Id.
    auto h1 = std::hash<rbmd::Id>{}(value.first);
    auto h2 = std::hash<rbmd::Id>{}(value.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

std::pair<rbmd::Id, rbmd::Id> OrderedPair(rbmd::Id lhs, rbmd::Id rhs) {
  if (lhs <= rhs) {
    return {lhs, rhs};
  }
  return {rhs, lhs};
}

std::string TrimCopy(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  return std::string(value);
}

std::string StripComment(std::string_view value) {
  auto trimmed = TrimCopy(value);
  auto pos = trimmed.find('#');
  if (pos != std::string_view::npos) {
    trimmed = trimmed.substr(0, pos);
  }
  return std::string(TrimCopy(trimmed));
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

std::string ToUpperCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return value;
}

rbmd::Real UnitBoltzmann(UNIT unit) {
  switch (unit) {
    case UNIT::LJ:
      return UnitFactor<UNIT::LJ>::_kb;
    case UNIT::METAL:
      return UnitFactor<UNIT::METAL>::_kb;
    case UNIT::REAL:
      return UnitFactor<UNIT::REAL>::_kb;
    default:
      throw std::invalid_argument("Unsupported unit style for velocity create");
  }
}

rbmd::Real UnitMvv2e(UNIT unit) {
  switch (unit) {
    case UNIT::LJ:
      return UnitFactor<UNIT::LJ>::_mvv2e;
    case UNIT::METAL:
      return UnitFactor<UNIT::METAL>::_mvv2e;
    case UNIT::REAL:
      return UnitFactor<UNIT::REAL>::_mvv2e;
    default:
      throw std::invalid_argument("Unsupported unit style for velocity create");
  }
}

std::uint64_t MixVelocitySeed(std::uint64_t seed, std::uint64_t atom_id) {
  std::uint64_t value =
      seed + 0x9e3779b97f4a7c15ULL + (atom_id << 6U) + (atom_id >> 2U);
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return value;
}

template <typename T>
std::string FormatCoeffSample(const T* values, std::size_t count,
                              std::size_t limit = 8) {
  if (values == nullptr || count == 0) {
    return "[]";
  }

  std::ostringstream oss;
  oss << '[';
  const std::size_t sample_count = std::min(count, limit);
  for (std::size_t i = 0; i < sample_count; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  if (count > sample_count) {
    oss << ", ...";
  }
  oss << ']';
  return oss.str();
}

template <typename MapType>
void ValidateCoeffCoverageOrThrow(const char* section_name,
                                  rbmd::Id expected_type_count,
                                  const MapType& coeffs) {
  if (section_name == nullptr || expected_type_count <= 0) {
    return;
  }

  std::vector<rbmd::Id> missing_type_ids;
  missing_type_ids.reserve(static_cast<std::size_t>(expected_type_count));
  for (rbmd::Id type = 0; type < expected_type_count; ++type) {
    if (coeffs.find(type) == coeffs.end()) {
      missing_type_ids.push_back(type + 1);
    }
  }

  if (missing_type_ids.empty()) {
    return;
  }

  std::ostringstream oss;
  oss << "LAMMPS data missing required " << section_name
      << " for declared type ids: ";
  for (std::size_t i = 0; i < missing_type_ids.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << missing_type_ids[i];
  }
  oss << " (expected_types=" << expected_type_count
      << ", parsed_entries=" << coeffs.size() << ")";
  throw std::runtime_error(oss.str());
}

void ValidateBondCoeffArrayOrThrow(const char* stage, const char* name,
                                   const rbmd::Real* values,
                                   std::size_t count) {
  if (stage == nullptr || name == nullptr || values == nullptr || count == 0) {
    return;
  }

  double max_value = 0.0;
  std::size_t max_index = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = values[i];
    if (!std::isfinite(value)) {
      std::ostringstream oss;
      oss << "[rank " << rbmd::mpi::CurrentRank()
          << "] LammpsFullReader observed non-finite " << name
          << " during " << stage
          << ", ptr=" << static_cast<const void*>(values)
          << ", count=" << count
          << ", sample=" << FormatCoeffSample(values, count);
      throw std::runtime_error(oss.str());
    }
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  if (std::string_view(name) == "bond_equilibrium" && max_value > 10.0) {
    std::ostringstream oss;
    oss << "[rank " << rbmd::mpi::CurrentRank()
        << "] LammpsFullReader observed suspicious " << name
        << " during " << stage
        << ": max=" << max_value << " at index " << max_index
        << ", ptr=" << static_cast<const void*>(values)
        << ", count=" << count
        << ", sample=" << FormatCoeffSample(values, count);
    throw std::runtime_error(oss.str());
  }
}

std::size_t CheckedMultiply(std::size_t lhs, std::size_t rhs,
                            std::string_view name) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error(std::string("LAMMPS replicate overflow for ") +
                              std::string(name));
  }
  return lhs * rhs;
}

rbmd::Id CheckedIdAdd(rbmd::Id lhs, std::size_t rhs,
                      std::string_view name) {
  const auto max_id = static_cast<std::size_t>(
      std::numeric_limits<rbmd::Id>::max());
  if (rhs > max_id || static_cast<std::size_t>(lhs) > max_id - rhs) {
    throw std::overflow_error(std::string("LAMMPS replicate id overflow for ") +
                              std::string(name));
  }
  return static_cast<rbmd::Id>(static_cast<std::size_t>(lhs) + rhs);
}

rbmd::Id NoMoleculeId() {
  return static_cast<rbmd::Id>(-1);
}

struct AtomLookup {
  std::unordered_map<rbmd::Id, std::size_t> index_by_id;
  std::size_t id_span = 0;
};

template <typename T>
std::size_t MaxIdSpan(const std::vector<T>& values, std::string_view name) {
  rbmd::Id max_id = 0;
  bool has_value = false;
  for (const auto& value : values) {
    if (value.id < 0) {
      throw std::runtime_error(std::string("LAMMPS replicate found negative ") +
                               std::string(name) + " id");
    }
    max_id = has_value ? std::max(max_id, value.id) : value.id;
    has_value = true;
  }
  if (!has_value) {
    return 0;
  }
  return static_cast<std::size_t>(CheckedIdAdd(max_id, 1, name));
}

AtomLookup BuildAtomLookup(const ParsedSnapshot& snapshot) {
  AtomLookup lookup;
  lookup.index_by_id.reserve(snapshot.atoms.size());
  for (std::size_t i = 0; i < snapshot.atoms.size(); ++i) {
    const auto id = snapshot.atoms[i].id;
    if (id < 0) {
      throw std::runtime_error("LAMMPS replicate found negative atom id");
    }
    if (!lookup.index_by_id.emplace(id, i).second) {
      throw std::runtime_error("LAMMPS replicate found duplicate atom id");
    }
    lookup.id_span = std::max(
        lookup.id_span,
        static_cast<std::size_t>(CheckedIdAdd(id, 1, "atom id span")));
  }
  return lookup;
}

const ParsedSnapshot::Atom& FindAtomOrThrow(const ParsedSnapshot& seed,
                                            const AtomLookup& lookup,
                                            rbmd::Id atom_id) {
  auto it = lookup.index_by_id.find(atom_id);
  if (it == lookup.index_by_id.end() || it->second >= seed.atoms.size()) {
    throw std::runtime_error(
        "LAMMPS replicate topology references an unknown atom id");
  }
  return seed.atoms[it->second];
}

int CheckedImageFlag(long long value) {
  if (value < static_cast<long long>(std::numeric_limits<int>::min()) ||
      value > static_cast<long long>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("LAMMPS replicate image flag overflow");
  }
  return static_cast<int>(value);
}

struct RemappedCoordinate {
  rbmd::Real value = 0.0;
  int image = 0;
};

RemappedCoordinate RemapPeriodic(rbmd::Real value, rbmd::Real lo,
                                 rbmd::Real hi) {
  const rbmd::Real length = hi - lo;
  if (length <= rbmd::Real{0}) {
    return {value, 0};
  }

  long long image = static_cast<long long>(
      std::floor(static_cast<double>((value - lo) / length)));
  rbmd::Real wrapped = value - static_cast<rbmd::Real>(image) * length;
  while (wrapped < lo) {
    wrapped += length;
    --image;
  }
  while (wrapped >= hi) {
    wrapped -= length;
    ++image;
  }
  return {wrapped, CheckedImageFlag(image)};
}

std::size_t ReplicaIndex(std::size_t ix, std::size_t iy, std::size_t iz,
                         std::size_t nx, std::size_t ny) {
  return iz * ny * nx + iy * nx + ix;
}

std::size_t WrapReplicaIndex(int value, std::size_t count) {
  if (count == 0) {
    return 0;
  }
  while (value < 0) {
    value += static_cast<int>(count);
  }
  while (value >= static_cast<int>(count)) {
    value -= static_cast<int>(count);
  }
  return static_cast<std::size_t>(value);
}

rbmd::Id SameReplicaAtomId(rbmd::Id atom_id, std::size_t rep_index,
                           std::size_t atom_id_span) {
  return CheckedIdAdd(atom_id,
                      CheckedMultiply(rep_index, atom_id_span,
                                      "replicated atom id offset"),
                      "replicated atom id");
}

rbmd::Id BondPeriodicAtomId(const ParsedSnapshot& seed,
                            const AtomLookup& lookup,
                            rbmd::Id anchor_atom,
                            rbmd::Id target_atom,
                            std::size_t ix,
                            std::size_t iy,
                            std::size_t iz,
                            std::size_t nx,
                            std::size_t ny,
                            std::size_t nz,
                            std::size_t atom_id_span) {
  const auto& anchor = FindAtomOrThrow(seed, lookup, anchor_atom);
  const auto& target = FindAtomOrThrow(seed, lookup, target_atom);
  const rbmd::Real prd[3] = {
      seed.box.xhi - seed.box.xlo,
      seed.box.yhi - seed.box.ylo,
      seed.box.zhi - seed.box.zlo};
  const rbmd::Real half[3] = {
      prd[0] * rbmd::Real{0.5},
      prd[1] * rbmd::Real{0.5},
      prd[2] * rbmd::Real{0.5}};
  const rbmd::Real center[3] = {
      (seed.box.xlo + seed.box.xhi) * rbmd::Real{0.5},
      (seed.box.ylo + seed.box.yhi) * rbmd::Real{0.5},
      (seed.box.zlo + seed.box.zhi) * rbmd::Real{0.5}};
  const rbmd::Real anchor_coord[3] = {anchor.x, anchor.y, anchor.z};
  const rbmd::Real target_coord[3] = {target.x, target.y, target.z};
  const std::size_t allrep[3] = {nx, ny, nz};
  const std::size_t thisrep[3] = {ix, iy, iz};
  std::size_t target_rep[3] = {ix, iy, iz};

  for (int dim = 0; dim < 3; ++dim) {
    int shift = 0;
    if (prd[dim] > rbmd::Real{0} &&
        std::fabs(anchor_coord[dim] - target_coord[dim]) > half[dim]) {
      shift = anchor_coord[dim] > center[dim] ? 1 : -1;
    }
    target_rep[dim] = WrapReplicaIndex(
        static_cast<int>(thisrep[dim]) + shift, allrep[dim]);
  }

  return SameReplicaAtomId(
      target_atom,
      ReplicaIndex(target_rep[0], target_rep[1], target_rep[2], nx, ny),
      atom_id_span);
}

ParsedSnapshot::Atom BuildReplicatedAtom(const ParsedSnapshot::Box& seed_box,
                                         const ParsedSnapshot::Box& output_box,
                                         const ParsedSnapshot::Atom& atom,
                                         std::size_t rep_index,
                                         std::size_t atom_id_span,
                                         std::size_t molecule_offset,
                                         rbmd::Real dx,
                                         rbmd::Real dy,
                                         rbmd::Real dz,
                                         bool use_bond_periodic) {
  auto replica = atom;
  replica.id = SameReplicaAtomId(atom.id, rep_index, atom_id_span);
  if (replica.molecule != NoMoleculeId()) {
    replica.molecule =
        CheckedIdAdd(replica.molecule, molecule_offset, "molecule id");
  }

  const rbmd::Real xprd = seed_box.xhi - seed_box.xlo;
  const rbmd::Real yprd = seed_box.yhi - seed_box.ylo;
  const rbmd::Real zprd = seed_box.zhi - seed_box.zlo;
  if (use_bond_periodic) {
    replica.x = atom.x + dx;
    replica.y = atom.y + dy;
    replica.z = atom.z + dz;
  } else {
    replica.x = atom.x + static_cast<rbmd::Real>(atom.ix) * xprd + dx;
    replica.y = atom.y + static_cast<rbmd::Real>(atom.iy) * yprd + dy;
    replica.z = atom.z + static_cast<rbmd::Real>(atom.iz) * zprd + dz;
  }

  const auto x = RemapPeriodic(replica.x, output_box.xlo, output_box.xhi);
  const auto y = RemapPeriodic(replica.y, output_box.ylo, output_box.yhi);
  const auto z = RemapPeriodic(replica.z, output_box.zlo, output_box.zhi);
  replica.x = x.value;
  replica.y = y.value;
  replica.z = z.value;
  replica.ix = x.image;
  replica.iy = y.image;
  replica.iz = z.image;
  return replica;
}

template <typename T>
T* AllocateTrackedArray(MDDataOwner& owner, std::size_t count) {
  if (count == 0) {
    return nullptr;
  }
  auto holder = std::shared_ptr<T>(static_cast<T*>(std::malloc(sizeof(T) * count)),
                                   [](T* ptr) { std::free(ptr); });
  if (!holder) {
    throw std::bad_alloc();
  }
  owner.adopt(std::shared_ptr<void>(holder, static_cast<void*>(holder.get())));
  return holder.get();
}

}  // namespace

LammpsFullReader::LammpsFullReader(std::shared_ptr<ISource> src,
                                   std::shared_ptr<IResourceLocator> locator,
                                   std::string file_path,
                                   std::shared_ptr<MDDataOwner> owner)
    : StructureReader(std::move(src), std::move(locator), std::move(owner),
                      std::move(file_path)) {
  LoadSpecialBondWeights();
  atom_style_ = DetectAtomStyle();
}
//TODO 从文件读取没有-1 所有id都得-1
int LammpsFullReader::ReadData() {
  streaming_ctx_ = nullptr;
  streaming_mode_ = false;

  auto storage = data();
  if (!storage) {
    throw std::runtime_error("LAMMPS reader requires MDData storage");
  }

  ResetSnapshot();
  LoadSpecialBondWeights();
  LoadReplicateFactors();
  LoadReplicateBondPeriodic();

  const bool mpi_enabled = mpi::IsEnabled();
  const int rank = mpi_enabled ? mpi::Rank() : 0;
  const bool replicate_enabled = ReplicateEnabled();

  atom_style_ = DetectAtomStyle();

  if (mpi_enabled && rank != 0) {
    rbmd::debug::StartupPhaseLog(
        "reader.lammps.non_root.before_begin_streaming");
    std::optional stream_ctx =
        mpi::BeginStreaming(snapshot_, static_cast<rbmd::Id>(0));
    rbmd::debug::StartupPhaseLog(
        "reader.lammps.non_root.after_begin_streaming");
    streaming_ctx_ = stream_ctx ? &*stream_ctx : nullptr;
    streaming_mode_ = stream_ctx.has_value();
    mpi::ReceiveLoop(*stream_ctx);
    streaming_ctx_ = nullptr;
    streaming_mode_ = false;
    mpi::EndStreaming(*stream_ctx);
    ApplyGaussianVelocitiesIfRequested();
    PopulateDataManagers();
    return 0;
  }

  std::optional<mpi::StreamCtx> stream_ctx;
  auto& cur = cursor();
  std::string_view line;
  bool header_complete = false;

  while (cur.nextLine(line)) {
    auto trimmed = TrimCopy(line);
    if (trimmed.empty()) {
      continue;
    }

    if (!header_complete) {
      auto section = DetectSection(trimmed);
      if (section != Section::None) {
        current_section_ = section;
        header_complete = true;
        if (mpi_enabled) {
          if (replicate_enabled) {
            AllocateDataSpace(expected_atoms_);
          } else {
            rbmd::debug::StartupPhaseLog(
                "reader.lammps.root.before_begin_streaming");
            stream_ctx = mpi::BeginStreaming(
                snapshot_, static_cast<rbmd::Id>(expected_atoms_));
            rbmd::debug::StartupPhaseLog(
                "reader.lammps.root.after_begin_streaming");
            streaming_ctx_ = stream_ctx ? &*stream_ctx : nullptr;
            streaming_mode_ = stream_ctx.has_value();
          }
        } else {
          AllocateDataSpace(expected_atoms_);
        }
        continue;
      }
      if (ParseHeader(trimmed) != 0) {
        return -1;
      }
      continue;
    }

    auto section = DetectSection(trimmed);
    if (section != Section::None) {
      current_section_ = section;
      continue;
    }

    if (IsComment(trimmed)) {
      continue;
    }

    switch (current_section_) {
      case Section::Masses:
        if (ParseMass(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::PairCoeffs:
        if (ParsePairCoeff(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::BondCoeffs:
        if (ParseBondCoeff(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::AngleCoeffs:
        if (ParseAngleCoeff(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::DihedralCoeffs:
        if (ParseDihedralCoeff(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::ImproperCoeffs:
        if (ParseImproperCoeff(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::Atoms:
        if (ParseAtom(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::Velocities:
        if (ParseVelocity(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::Bonds:
        if (ParseBond(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::Angles:
        if (ParseAngle(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::Dihedrals:
        if (ParseDihedral(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::Impropers:
        if (ParseImproper(trimmed) != 0) {
          return -1;
        }
        break;
      case Section::None:
        break;
    }
  }

  if (replicate_enabled) {
    if (mpi_enabled) {
      StreamReplicatedSnapshot();
    } else {
      ApplyReplicate();
    }
  } else if (mpi_enabled && !stream_ctx) {
    rbmd::debug::StartupPhaseLog(
        "reader.lammps.root.late_before_begin_streaming");
    stream_ctx = mpi::BeginStreaming(
        snapshot_, static_cast<rbmd::Id>(expected_atoms_));
    rbmd::debug::StartupPhaseLog(
        "reader.lammps.root.late_after_begin_streaming");
  }

  if (mpi_enabled && stream_ctx) {
    mpi::EndStreaming(*stream_ctx);
  }
  streaming_ctx_ = nullptr;
  streaming_mode_ = false;

  ApplyGaussianVelocitiesIfRequested();
  PopulateDataManagers();

  return 0;
}

void LammpsFullReader::AllocateDataSpace(std::size_t num_atoms) {
  snapshot_.atoms.clear();
  snapshot_.atoms.reserve(num_atoms);
  atom_index_by_id_.clear();
  atom_index_by_id_.reserve(num_atoms);
  snapshot_.bonds.clear();
  snapshot_.bonds.reserve(expected_bonds_);
  snapshot_.angles.clear();
  snapshot_.angles.reserve(expected_angles_);
  snapshot_.dihedrals.clear();
  snapshot_.dihedrals.reserve(expected_dihedrals_);
  snapshot_.impropers.clear();
  snapshot_.impropers.reserve(expected_impropers_);
}

void LammpsFullReader::LoadReplicateFactors() {
  replicate_factors_ = {1, 1, 1};

  auto config = DataManager::getInstance().getConfigData();
  if (!config) {
    return;
  }

  const bool has_replicate =
      config->PathExists({"init_configuration", "read_data", "replicate"});
  const bool has_replicate_dims =
      config->PathExists({"init_configuration", "read_data", "replicate_dims"});
  if (!has_replicate && !has_replicate_dims) {
    return;
  }

  const auto key = has_replicate ? "replicate" : "replicate_dims";
  auto values = config->GetArray<int>(key, "init_configuration", "read_data");
  if (values.size() != 3) {
    throw std::runtime_error(
        "init_configuration.read_data.replicate must contain exactly 3 integers");
  }

  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i] <= 0) {
      throw std::runtime_error(
          "init_configuration.read_data.replicate values must be positive");
    }
    replicate_factors_[i] = static_cast<std::size_t>(values[i]);
  }
}

void LammpsFullReader::LoadReplicateBondPeriodic() {
  replicate_bond_periodic_ = false;

  auto config = DataManager::getInstance().getConfigData();
  if (!config) {
    return;
  }

  if (config->PathExists(
          {"init_configuration", "read_data", "replicate_bond_periodic"})) {
    replicate_bond_periodic_ =
        config->Get<bool>("replicate_bond_periodic", "init_configuration",
                          "read_data");
    return;
  }

  if (config->PathExists({"init_configuration", "read_data", "bond_periodic"})) {
    replicate_bond_periodic_ =
        config->Get<bool>("bond_periodic", "init_configuration", "read_data");
    return;
  }

  if (config->PathExists({"init_configuration", "read_data", "bond/periodic"})) {
    replicate_bond_periodic_ =
        config->Get<bool>("bond/periodic", "init_configuration", "read_data");
  }
}

void LammpsFullReader::ApplyGaussianVelocitiesIfRequested() {
  auto config = DataManager::getInstance().getConfigData();
  if (!config || !config->PathExists(
                     {"init_configuration", "read_data", "velocity_type"})) {
    return;
  }

  auto velocity_type = ToUpperCopy(config->Get<std::string>(
      "velocity_type", "init_configuration", "read_data"));
  if (velocity_type.empty() || velocity_type == "NONE") {
    return;
  }
  if (velocity_type != "GAUSS" && velocity_type != "GAUSSIAN") {
    throw std::runtime_error("Unsupported velocity_type: " + velocity_type);
  }

  int local_has_velocity_data = parsed_velocity_count_ > 0 ? 1 : 0;
  int global_has_velocity_data = local_has_velocity_data;
#ifdef READER_ENABLE_MPI
  if (mpi::IsEnabled()) {
    MPI_Allreduce(&local_has_velocity_data, &global_has_velocity_data, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }
#endif
  if (global_has_velocity_data) {
    return;
  }

  auto temperature = config->GetArray<rbmd::Real>("temperature", "execution");
  if (temperature.empty() || !std::isfinite(temperature[0]) ||
      temperature[0] <= rbmd::Real{0}) {
    throw std::runtime_error(
        "velocity_type GAUSS requires execution.temperature[0] > 0");
  }
  const rbmd::Real target_temperature = temperature[0];

  std::string unit_name = "LJ";
  if (config->PathExists({"init_configuration", "read_data", "unit"})) {
    unit_name =
        config->Get<std::string>("unit", "init_configuration", "read_data");
  }
  const UNIT unit = ParseUnit(unit_name);
  const double kb = static_cast<double>(UnitBoltzmann(unit));
  const double mvv2e = static_cast<double>(UnitMvv2e(unit));

  int seed = 12345;
  if (config->PathExists({"init_configuration", "read_data", "velocity_seed"})) {
    seed = config->Get<int>("velocity_seed", "init_configuration", "read_data");
  } else if (config->PathExists({"init_configuration", "read_data", "seed"})) {
    seed = config->Get<int>("seed", "init_configuration", "read_data");
  }
  if (seed <= 0) {
    throw std::runtime_error("velocity_type GAUSS requires a positive seed");
  }

  auto atom_mass = [this](rbmd::Id type) -> double {
    const auto it = snapshot_.masses.find(type);
    if (it == snapshot_.masses.end() || !std::isfinite(it->second) ||
        it->second <= rbmd::Real{0}) {
      throw std::runtime_error(
          "velocity_type GAUSS requires positive mass for every atom type");
    }
    return static_cast<double>(it->second);
  };

  double local_mass = 0.0;
  double local_mvx = 0.0;
  double local_mvy = 0.0;
  double local_mvz = 0.0;
  for (auto& atom : snapshot_.atoms) {
    const double mass = atom_mass(atom.type);
    std::mt19937_64 rng(MixVelocitySeed(static_cast<std::uint64_t>(seed),
                                        static_cast<std::uint64_t>(atom.id + 1)));
    std::normal_distribution<double> gaussian(0.0, 1.0);
    const double factor = 1.0 / std::sqrt(mass);
    atom.vx = static_cast<rbmd::Real>(gaussian(rng) * factor);
    atom.vy = static_cast<rbmd::Real>(gaussian(rng) * factor);
    atom.vz = static_cast<rbmd::Real>(gaussian(rng) * factor);
    local_mass += mass;
    local_mvx += mass * static_cast<double>(atom.vx);
    local_mvy += mass * static_cast<double>(atom.vy);
    local_mvz += mass * static_cast<double>(atom.vz);
  }

  double global_mass = local_mass;
  double global_mvx = local_mvx;
  double global_mvy = local_mvy;
  double global_mvz = local_mvz;
  long long local_atoms = static_cast<long long>(snapshot_.atoms.size());
  long long global_atoms = local_atoms;
#ifdef READER_ENABLE_MPI
  if (mpi::IsEnabled()) {
    double local_momentum[4] = {local_mass, local_mvx, local_mvy, local_mvz};
    double global_momentum[4] = {};
    MPI_Allreduce(local_momentum, global_momentum, 4, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    global_mass = global_momentum[0];
    global_mvx = global_momentum[1];
    global_mvy = global_momentum[2];
    global_mvz = global_momentum[3];
    MPI_Allreduce(&local_atoms, &global_atoms, 1, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif
  if (global_atoms <= 0 || global_mass <= 0.0) {
    return;
  }

  const double bias_vx = global_mvx / global_mass;
  const double bias_vy = global_mvy / global_mass;
  const double bias_vz = global_mvz / global_mass;
  double local_temp_sum = 0.0;
  for (auto& atom : snapshot_.atoms) {
    atom.vx = static_cast<rbmd::Real>(static_cast<double>(atom.vx) - bias_vx);
    atom.vy = static_cast<rbmd::Real>(static_cast<double>(atom.vy) - bias_vy);
    atom.vz = static_cast<rbmd::Real>(static_cast<double>(atom.vz) - bias_vz);
    const double mass = atom_mass(atom.type);
    local_temp_sum += mass *
                      (static_cast<double>(atom.vx) * atom.vx +
                       static_cast<double>(atom.vy) * atom.vy +
                       static_cast<double>(atom.vz) * atom.vz);
  }
  local_temp_sum *= mvv2e;

  double global_temp_sum = local_temp_sum;
#ifdef READER_ENABLE_MPI
  if (mpi::IsEnabled()) {
    MPI_Allreduce(&local_temp_sum, &global_temp_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif
  if (global_temp_sum <= 0.0 || !std::isfinite(global_temp_sum)) {
    throw std::runtime_error("velocity_type GAUSS failed to create finite kinetic energy");
  }

  const bool use_shake =
      config->GetJudge<bool>("fix_shake", "hyper_parameters", "extend");
  long long dof = 3LL * global_atoms - 3LL;
  if (dof <= 0) {
    throw std::runtime_error("velocity_type GAUSS computed non-positive temperature DOF");
  }

  const double scale =
      std::sqrt(static_cast<double>(target_temperature) *
                static_cast<double>(dof) * kb / global_temp_sum);
  for (auto& atom : snapshot_.atoms) {
    atom.vx = static_cast<rbmd::Real>(static_cast<double>(atom.vx) * scale);
    atom.vy = static_cast<rbmd::Real>(static_cast<double>(atom.vy) * scale);
    atom.vz = static_cast<rbmd::Real>(static_cast<double>(atom.vz) * scale);
  }

  std::ostringstream oss;
  oss << "type=GAUSS target_temperature=" << target_temperature
      << " seed=" << seed << " global_atoms=" << global_atoms
      << " dof=" << dof << " unit=" << unit_name
      << " fix_shake=" << (use_shake ? "true" : "false");
  rbmd::debug::StartupPhaseLog("reader.lammps.velocity_create", oss.str());
}

bool LammpsFullReader::ReplicateEnabled() const {
  return replicate_factors_[0] != 1 || replicate_factors_[1] != 1 ||
         replicate_factors_[2] != 1;
}

std::size_t LammpsFullReader::ReplicateCount() const {
  return CheckedMultiply(
      CheckedMultiply(replicate_factors_[0], replicate_factors_[1],
                      "replicate xy count"),
      replicate_factors_[2], "replicate xyz count");
}

bool LammpsFullReader::ReplicateRequiresImageFlags() const {
  return replicate_bond_periodic_ || atom_style_ == AtomStyle::Full ||
         expected_bonds_ > 0 || expected_angles_ > 0 ||
         expected_dihedrals_ > 0 || expected_impropers_ > 0;
}

void LammpsFullReader::ValidateReplicateImageFlagsOrThrow() const {
  if (!ReplicateEnabled() || !ReplicateRequiresImageFlags()) {
    return;
  }
  if (!atom_image_flags_seen_ || atom_image_flags_missing_) {
    throw std::runtime_error(
        "LAMMPS replicate for molecular/topology data requires image flags "
        "(ix iy iz) on every Atoms entry");
  }
}

void LammpsFullReader::ApplyReplicate() {
  if (!ReplicateEnabled()) {
    return;
  }
  if (triclinic_box_detected_) {
    throw std::runtime_error(
        "LAMMPS replicate currently supports only orthogonal boxes; "
        "triclinic xy xz yz tilt factors are not supported");
  }
  ValidateReplicateImageFlagsOrThrow();

  const ParsedSnapshot seed = snapshot_;
  const std::size_t nrep = ReplicateCount();
  const AtomLookup atom_lookup = BuildAtomLookup(seed);
  const std::size_t atom_id_span = atom_lookup.id_span;
  const std::size_t base_atoms = seed.atoms.size();
  const std::size_t base_bonds = seed.bonds.size();
  const std::size_t base_angles = seed.angles.size();
  const std::size_t base_dihedrals = seed.dihedrals.size();
  const std::size_t base_impropers = seed.impropers.size();
  const std::size_t bond_id_span = MaxIdSpan(seed.bonds, "bond id span");
  const std::size_t angle_id_span = MaxIdSpan(seed.angles, "angle id span");
  const std::size_t dihedral_id_span =
      MaxIdSpan(seed.dihedrals, "dihedral id span");
  const std::size_t improper_id_span =
      MaxIdSpan(seed.impropers, "improper id span");
  const std::size_t total_atoms =
      CheckedMultiply(base_atoms, nrep, "atom count");
  const std::size_t total_bonds =
      CheckedMultiply(base_bonds, nrep, "bond count");
  const std::size_t total_angles =
      CheckedMultiply(base_angles, nrep, "angle count");
  const std::size_t total_dihedrals =
      CheckedMultiply(base_dihedrals, nrep, "dihedral count");
  const std::size_t total_impropers =
      CheckedMultiply(base_impropers, nrep, "improper count");

  snapshot_ = seed;
  snapshot_.atoms.clear();
  snapshot_.bonds.clear();
  snapshot_.angles.clear();
  snapshot_.dihedrals.clear();
  snapshot_.impropers.clear();
  snapshot_.atoms.reserve(total_atoms);
  snapshot_.bonds.reserve(total_bonds);
  snapshot_.angles.reserve(total_angles);
  snapshot_.dihedrals.reserve(total_dihedrals);
  snapshot_.impropers.reserve(total_impropers);

  const auto nx = replicate_factors_[0];
  const auto ny = replicate_factors_[1];
  const auto nz = replicate_factors_[2];
  const rbmd::Real xprd = seed.box.xhi - seed.box.xlo;
  const rbmd::Real yprd = seed.box.yhi - seed.box.ylo;
  const rbmd::Real zprd = seed.box.zhi - seed.box.zlo;
  snapshot_.box.xhi = seed.box.xlo + static_cast<rbmd::Real>(nx) * xprd;
  snapshot_.box.yhi = seed.box.ylo + static_cast<rbmd::Real>(ny) * yprd;
  snapshot_.box.zhi = seed.box.zlo + static_cast<rbmd::Real>(nz) * zprd;

  rbmd::Id molecule_span = 0;
  if (atom_style_ == AtomStyle::Full) {
    for (const auto& atom : seed.atoms) {
      if (atom.molecule != NoMoleculeId()) {
        molecule_span = std::max(molecule_span, CheckedIdAdd(atom.molecule, 1,
                                                             "molecule span"));
      }
    }
  }

  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        const std::size_t rep_index = iz * ny * nx + iy * nx + ix;
        const std::size_t molecule_offset =
            rep_index * static_cast<std::size_t>(molecule_span);
        const rbmd::Real dx = static_cast<rbmd::Real>(ix) * xprd;
        const rbmd::Real dy = static_cast<rbmd::Real>(iy) * yprd;
        const rbmd::Real dz = static_cast<rbmd::Real>(iz) * zprd;

        for (const auto& atom : seed.atoms) {
          snapshot_.atoms.push_back(BuildReplicatedAtom(
              seed.box, snapshot_.box, atom, rep_index, atom_id_span,
              molecule_offset, dx, dy, dz, replicate_bond_periodic_));
        }

        const auto same_replica_atom = [&](rbmd::Id atom_id) {
          return SameReplicaAtomId(atom_id, rep_index, atom_id_span);
        };
        const auto topology_atom = [&](rbmd::Id anchor, rbmd::Id target) {
          if (!replicate_bond_periodic_) {
            return same_replica_atom(target);
          }
          return BondPeriodicAtomId(seed, atom_lookup, anchor, target, ix, iy,
                                    iz, nx, ny, nz, atom_id_span);
        };

        for (const auto& bond : seed.bonds) {
          auto replica = bond;
          replica.id = CheckedIdAdd(
              bond.id,
              CheckedMultiply(rep_index, bond_id_span, "bond id offset"),
              "bond id");
          replica.atom1 = same_replica_atom(bond.atom1);
          replica.atom2 = topology_atom(bond.atom1, bond.atom2);
          snapshot_.bonds.push_back(replica);
        }
        for (const auto& angle : seed.angles) {
          auto replica = angle;
          replica.id = CheckedIdAdd(
              angle.id,
              CheckedMultiply(rep_index, angle_id_span, "angle id offset"),
              "angle id");
          replica.atom1 = topology_atom(angle.atom2, angle.atom1);
          replica.atom2 = same_replica_atom(angle.atom2);
          replica.atom3 = topology_atom(angle.atom2, angle.atom3);
          snapshot_.angles.push_back(replica);
        }
        for (const auto& dihedral : seed.dihedrals) {
          auto replica = dihedral;
          replica.id = CheckedIdAdd(
              dihedral.id,
              CheckedMultiply(rep_index, dihedral_id_span,
                              "dihedral id offset"),
              "dihedral id");
          replica.atom1 = topology_atom(dihedral.atom2, dihedral.atom1);
          replica.atom2 = same_replica_atom(dihedral.atom2);
          replica.atom3 = topology_atom(dihedral.atom2, dihedral.atom3);
          replica.atom4 = topology_atom(dihedral.atom2, dihedral.atom4);
          snapshot_.dihedrals.push_back(replica);
        }
        for (const auto& improper : seed.impropers) {
          auto replica = improper;
          replica.id = CheckedIdAdd(
              improper.id,
              CheckedMultiply(rep_index, improper_id_span,
                              "improper id offset"),
              "improper id");
          replica.atom1 = topology_atom(improper.atom2, improper.atom1);
          replica.atom2 = same_replica_atom(improper.atom2);
          replica.atom3 = topology_atom(improper.atom2, improper.atom3);
          replica.atom4 = topology_atom(improper.atom2, improper.atom4);
          snapshot_.impropers.push_back(replica);
        }
      }
    }
  }

  expected_atoms_ = total_atoms;
  expected_bonds_ = total_bonds;
  expected_angles_ = total_angles;
  expected_dihedrals_ = total_dihedrals;
  expected_impropers_ = total_impropers;
}

void LammpsFullReader::StreamReplicatedSnapshot() {
  if (triclinic_box_detected_) {
    throw std::runtime_error(
        "LAMMPS replicate currently supports only orthogonal boxes; "
        "triclinic xy xz yz tilt factors are not supported");
  }
  ValidateReplicateImageFlagsOrThrow();

  const ParsedSnapshot seed = snapshot_;
  const std::size_t nrep = ReplicateCount();
  const AtomLookup atom_lookup = BuildAtomLookup(seed);
  const std::size_t atom_id_span = atom_lookup.id_span;
  const std::size_t base_atoms = seed.atoms.size();
  const std::size_t base_bonds = seed.bonds.size();
  const std::size_t base_angles = seed.angles.size();
  const std::size_t base_dihedrals = seed.dihedrals.size();
  const std::size_t base_impropers = seed.impropers.size();
  const std::size_t bond_id_span = MaxIdSpan(seed.bonds, "bond id span");
  const std::size_t angle_id_span = MaxIdSpan(seed.angles, "angle id span");
  const std::size_t dihedral_id_span =
      MaxIdSpan(seed.dihedrals, "dihedral id span");
  const std::size_t improper_id_span =
      MaxIdSpan(seed.impropers, "improper id span");

  expected_atoms_ = CheckedMultiply(base_atoms, nrep, "atom count");
  expected_bonds_ = CheckedMultiply(base_bonds, nrep, "bond count");
  expected_angles_ = CheckedMultiply(base_angles, nrep, "angle count");
  expected_dihedrals_ =
      CheckedMultiply(base_dihedrals, nrep, "dihedral count");
  expected_impropers_ =
      CheckedMultiply(base_impropers, nrep, "improper count");

  const auto nx = replicate_factors_[0];
  const auto ny = replicate_factors_[1];
  const auto nz = replicate_factors_[2];
  const rbmd::Real xprd = seed.box.xhi - seed.box.xlo;
  const rbmd::Real yprd = seed.box.yhi - seed.box.ylo;
  const rbmd::Real zprd = seed.box.zhi - seed.box.zlo;

  snapshot_ = seed;
  snapshot_.box.xhi = seed.box.xlo + static_cast<rbmd::Real>(nx) * xprd;
  snapshot_.box.yhi = seed.box.ylo + static_cast<rbmd::Real>(ny) * yprd;
  snapshot_.box.zhi = seed.box.zlo + static_cast<rbmd::Real>(nz) * zprd;
  snapshot_.atoms.clear();
  snapshot_.bonds.clear();
  snapshot_.angles.clear();
  snapshot_.dihedrals.clear();
  snapshot_.impropers.clear();

  std::optional<mpi::StreamCtx> stream_ctx =
      mpi::BeginStreaming(snapshot_, static_cast<rbmd::Id>(expected_atoms_));
  streaming_ctx_ = stream_ctx ? &*stream_ctx : nullptr;
  streaming_mode_ = stream_ctx.has_value();

  if (!stream_ctx) {
    return;
  }

  rbmd::Id molecule_span = 0;
  if (atom_style_ == AtomStyle::Full) {
    for (const auto& atom : seed.atoms) {
      if (atom.molecule != NoMoleculeId()) {
        molecule_span = std::max(molecule_span, CheckedIdAdd(atom.molecule, 1,
                                                             "molecule span"));
      }
    }
  }

  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        const std::size_t rep_index = iz * ny * nx + iy * nx + ix;
        const std::size_t molecule_offset =
            rep_index * static_cast<std::size_t>(molecule_span);
        const rbmd::Real dx = static_cast<rbmd::Real>(ix) * xprd;
        const rbmd::Real dy = static_cast<rbmd::Real>(iy) * yprd;
        const rbmd::Real dz = static_cast<rbmd::Real>(iz) * zprd;

        for (const auto& atom : seed.atoms) {
          auto replica = BuildReplicatedAtom(
              seed.box, snapshot_.box, atom, rep_index, atom_id_span,
              molecule_offset, dx, dy, dz, replicate_bond_periodic_);
          mpi::StreamAtom(*stream_ctx, replica);
        }
      }
    }
  }

  auto for_each_replica = [&](const auto& handler) {
    for (std::size_t iz = 0; iz < nz; ++iz) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        for (std::size_t ix = 0; ix < nx; ++ix) {
          const std::size_t rep_index = iz * ny * nx + iy * nx + ix;
          const auto same_replica_atom = [&](rbmd::Id atom_id) {
            return SameReplicaAtomId(atom_id, rep_index, atom_id_span);
          };
          const auto topology_atom = [&](rbmd::Id anchor, rbmd::Id target) {
            if (!replicate_bond_periodic_) {
              return same_replica_atom(target);
            }
            return BondPeriodicAtomId(seed, atom_lookup, anchor, target, ix, iy,
                                      iz, nx, ny, nz, atom_id_span);
          };

          handler(rep_index, same_replica_atom, topology_atom);
        }
      }
    }
  };

  for_each_replica([&](std::size_t rep_index, const auto& same_replica_atom,
                       const auto& topology_atom) {
    for (const auto& bond : seed.bonds) {
      auto replica = bond;
      replica.id = CheckedIdAdd(
          bond.id, CheckedMultiply(rep_index, bond_id_span, "bond id offset"),
          "bond id");
      replica.atom1 = same_replica_atom(bond.atom1);
      replica.atom2 = topology_atom(bond.atom1, bond.atom2);
      mpi::StreamBond(*stream_ctx, replica);
    }
  });

  for_each_replica([&](std::size_t rep_index, const auto& same_replica_atom,
                       const auto& topology_atom) {
    for (const auto& angle : seed.angles) {
      auto replica = angle;
      replica.id = CheckedIdAdd(
          angle.id,
          CheckedMultiply(rep_index, angle_id_span, "angle id offset"),
          "angle id");
      replica.atom1 = topology_atom(angle.atom2, angle.atom1);
      replica.atom2 = same_replica_atom(angle.atom2);
      replica.atom3 = topology_atom(angle.atom2, angle.atom3);
      mpi::StreamAngle(*stream_ctx, replica);
    }
  });

  for_each_replica([&](std::size_t rep_index, const auto& same_replica_atom,
                       const auto& topology_atom) {
    for (const auto& dihedral : seed.dihedrals) {
      auto replica = dihedral;
      replica.id = CheckedIdAdd(
          dihedral.id,
          CheckedMultiply(rep_index, dihedral_id_span, "dihedral id offset"),
          "dihedral id");
      replica.atom1 = topology_atom(dihedral.atom2, dihedral.atom1);
      replica.atom2 = same_replica_atom(dihedral.atom2);
      replica.atom3 = topology_atom(dihedral.atom2, dihedral.atom3);
      replica.atom4 = topology_atom(dihedral.atom2, dihedral.atom4);
      mpi::StreamDihedral(*stream_ctx, replica);
    }
  });

  for_each_replica([&](std::size_t rep_index, const auto& same_replica_atom,
                       const auto& topology_atom) {
    for (const auto& improper : seed.impropers) {
      auto replica = improper;
      replica.id = CheckedIdAdd(
          improper.id,
          CheckedMultiply(rep_index, improper_id_span, "improper id offset"),
          "improper id");
      replica.atom1 = topology_atom(improper.atom2, improper.atom1);
      replica.atom2 = same_replica_atom(improper.atom2);
      replica.atom3 = topology_atom(improper.atom2, improper.atom3);
      replica.atom4 = topology_atom(improper.atom2, improper.atom4);
      mpi::StreamImproper(*stream_ctx, replica);
    }
  });

  mpi::EndStreaming(*stream_ctx);
  streaming_ctx_ = nullptr;
  streaming_mode_ = false;
}

int LammpsFullReader::ParseHeader(std::string_view line) {
  auto text = std::string(TrimCopy(line));
  if (text.empty()) {
    return 0;
  }

  std::istringstream iss(text);
  double value1 = 0.0;
  double value2 = 0.0;

  if (EndsWith(text, "atoms")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    expected_atoms_ = static_cast<std::size_t>(value1);
    return 0;
  }
  if (EndsWith(text, "bonds")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    expected_bonds_ = static_cast<std::size_t>(value1);
    return 0;
  }
  if (EndsWith(text, "angles")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    expected_angles_ = static_cast<std::size_t>(value1);
    return 0;
  }
  if (EndsWith(text, "dihedrals")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    expected_dihedrals_ = static_cast<std::size_t>(value1);
    return 0;
  }
  if (EndsWith(text, "impropers")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    expected_impropers_ = static_cast<std::size_t>(value1);
    return 0;
  }
  if (EndsWith(text, "atom types")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.topology.atom_types = static_cast<rbmd::Id>(value1);
    return 0;
  }
  if (EndsWith(text, "bond types")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.topology.bond_types = static_cast<rbmd::Id>(value1);
    return 0;
  }
  if (EndsWith(text, "angle types")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.topology.angle_types = static_cast<rbmd::Id>(value1);
    return 0;
  }
  if (EndsWith(text, "dihedral types")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.topology.dihedral_types = static_cast<rbmd::Id>(value1);
    return 0;
  }
  if (EndsWith(text, "improper types")) {
    iss >> value1;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.topology.improper_types = static_cast<rbmd::Id>(value1);
    return 0;
  }

  if (text.find("xlo") != std::string::npos &&
      text.find("xhi") != std::string::npos) {
    iss >> value1 >> value2;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.box.xlo = value1;
    snapshot_.box.xhi = value2;
    return 0;
  }
  if (text.find("ylo") != std::string::npos &&
      text.find("yhi") != std::string::npos) {
    iss >> value1 >> value2;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.box.ylo = value1;
    snapshot_.box.yhi = value2;
    return 0;
  }
  if (text.find("zlo") != std::string::npos &&
      text.find("zhi") != std::string::npos) {
    iss >> value1 >> value2;
    if (iss.fail()) {
      return -1;
    }
    snapshot_.box.zlo = value1;
    snapshot_.box.zhi = value2;
    return 0;
  }
  if (text.find("xy") != std::string::npos &&
      text.find("xz") != std::string::npos &&
      text.find("yz") != std::string::npos) {
    triclinic_box_detected_ = true;
    return 0;
  }

  return 0;
}

int LammpsFullReader::ParseMass(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  rbmd::Id type = 0;
  rbmd::Real mass = 0.0;
  iss >> type >> mass;
  if (iss.fail() || type <= 0) {
    return -1;
  }
  type -= 1;
  snapshot_.masses[type] = mass;
  return 0;
}

int LammpsFullReader::ParsePairCoeff(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::PairCoeff coeff;
  iss >> coeff.type;
  if (iss.fail() || coeff.type <= 0) {
    return -1;
  }

  if (IsEamForceField()) {
    std::string potential_file;
    iss >> potential_file;
    if (iss.fail() || potential_file.empty()) {
      return -1;
    }
    if (eam_potential_file_.empty()) {
      eam_potential_file_ = potential_file;
    }
    coeff.type -= 1;
    snapshot_.pair_coeffs[coeff.type] = coeff;
    return 0;
  }

  iss >> coeff.epsilon >> coeff.sigma;
  if (iss.fail()) {
    return -1;
  }
  coeff.type -= 1;
  snapshot_.pair_coeffs[coeff.type] = coeff;
  return 0;
}

int LammpsFullReader::ParseBondCoeff(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::BondCoeff coeff;
  iss >> coeff.type >> coeff.k >> coeff.equilibrium;
  if (iss.fail() || coeff.type <= 0) {
    return -1;
  }
  coeff.type -= 1;
  snapshot_.bond_coeffs[coeff.type] = coeff;
  return 0;
}

int LammpsFullReader::ParseAngleCoeff(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::AngleCoeff coeff;
  iss >> coeff.type >> coeff.k >> coeff.equilibrium;
  if (iss.fail() || coeff.type <= 0) {
    return -1;
  }
  coeff.type -= 1;
  snapshot_.angle_coeffs[coeff.type] = coeff;
  return 0;
}

int LammpsFullReader::ParseDihedralCoeff(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::DihedralCoeff coeff;
  auto dihedral_type = DataManager::getInstance().getConfigData()->Get<std::string>("dihedral_type", "hyper_parameters", "force_field");
  if (dihedral_type == "harmonic") {
    iss >> coeff.type >> coeff.k >> coeff.sign >> coeff.multiplicity;
    if (iss.fail() || coeff.type <= 0) {
      return -1;
    }
    coeff.type -= 1;
  }
  else if (dihedral_type == "opls") {
    iss >> coeff.type >> coeff.k1 >> coeff.k2 >> coeff.k3 >> coeff.k4;
    if (iss.fail() || coeff.type <= 0) {
      return -1;
    }
    coeff.type -= 1;
  }

  snapshot_.dihedral_coeffs[coeff.type] = coeff;
  return 0;
}

int LammpsFullReader::ParseImproperCoeff(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::ImproperCoeff coeff;
  auto improper_type = DataManager::getInstance().getConfigData()->Get
    <std::string>("improper_type", "hyper_parameters", "force_field");
  if (improper_type == "harmonic") {
    iss >> coeff.type >> coeff.k >> coeff.degree;
    if (iss.fail() || coeff.type <= 0) {
      return -1;
    }
    coeff.type -= 1;
  }else if (improper_type == "cvff") {
    iss >> coeff.type >> coeff.k >> coeff.d >> coeff.n;
    if (iss.fail() || coeff.type <= 0) {
      return -1;
    }
    coeff.type -= 1;
  }
  snapshot_.improper_coeffs[coeff.type] = coeff;
  return 0;
}

int LammpsFullReader::ParseAtom(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::Atom atom;
  atom.molecule = 0;
  atom.charge = 0.0;
  atom.vx = atom.vy = atom.vz = 0.0;

  switch (atom_style_) {
    case AtomStyle::Atomic:
      iss >> atom.id >> atom.type >> atom.x >> atom.y >> atom.z;
      break;
    case AtomStyle::Charge:
      iss >> atom.id >> atom.type >> atom.charge >> atom.x >> atom.y >> atom.z;
      break;
    case AtomStyle::Full:
    default:
      iss >> atom.id >> atom.molecule >> atom.type >> atom.charge >> atom.x >>
          atom.y >> atom.z;
      break;
  }
  if (iss.fail()) {
    return -1;
  }

  int image_x = 0;
  int image_y = 0;
  int image_z = 0;
  if (iss >> image_x) {
    if (!(iss >> image_y >> image_z)) {
      return -1;
    }
    atom.ix = image_x;
    atom.iy = image_y;
    atom.iz = image_z;
    atom_image_flags_seen_ = true;
  } else {
    atom_image_flags_missing_ = true;
    iss.clear();
  }

  if (atom.id <= 0) {
    return -1;
  }
  atom.id -= 1;
  if (atom.type <= 0) {
    return -1;
  }
  atom.type -= 1;
  if (atom_style_ == AtomStyle::Full) {
    atom.molecule = atom.molecule > 0 ? atom.molecule - 1 : -1;
  }
  if (streaming_mode_ && streaming_ctx_) {
    mpi::StreamAtom(*streaming_ctx_, atom);
    return 0;
  }
  if (snapshot_.atoms.size() >= expected_atoms_) {
    return -1;
  }
  const auto index = snapshot_.atoms.size();
  if (!atom_index_by_id_.emplace(atom.id, index).second) {
    return -1;
  }
  snapshot_.atoms.push_back(atom);
  return 0;
}

int LammpsFullReader::ParseVelocity(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  rbmd::Id id = 0;
  rbmd::Real vx = 0.0;
  rbmd::Real vy = 0.0;
  rbmd::Real vz = 0.0;
  iss >> id >> vx >> vy >> vz;
  if (iss.fail() || id <= 0) {
    return -1;
  }
  ++parsed_velocity_count_;
  id -= 1;
  if (streaming_mode_ && streaming_ctx_) {
    mpi::StreamVelocity(*streaming_ctx_, id, vx, vy, vz);
    return 0;
  }
  const auto it = atom_index_by_id_.find(id);
  if (it == atom_index_by_id_.end() || it->second >= snapshot_.atoms.size()) {
    return -1;
  }
  auto& atom = snapshot_.atoms[it->second];
  atom.vx = vx;
  atom.vy = vy;
  atom.vz = vz;
  return 0;
}

int LammpsFullReader::ParseBond(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::Bond bond;
  iss >> bond.id >> bond.type >> bond.atom1 >> bond.atom2;
  if (iss.fail()) {
    return -1;
  }
  if (bond.id <= 0 || bond.type <= 0 || bond.atom1 <= 0 || bond.atom2 <= 0) {
    return -1;
  }
  bond.id -= 1;
  bond.type -= 1;
  bond.atom1 -= 1;
  bond.atom2 -= 1;
  if (streaming_mode_ && streaming_ctx_) {
    mpi::StreamBond(*streaming_ctx_, bond);
    return 0;
  }
  snapshot_.bonds.push_back(bond);
  if (!mpi::IsEnabled()) {
    bond_adjacency_.emplace(bond.atom1, bond.atom2);
    bond_adjacency_.emplace(bond.atom2, bond.atom1);
  }
  return 0;
}

int LammpsFullReader::ParseAngle(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::Angle angle;
  iss >> angle.id >> angle.type >> angle.atom1 >> angle.atom2 >> angle.atom3;
  if (iss.fail()) {
    return -1;
  }
  if (angle.id <= 0 || angle.type <= 0 || angle.atom1 <= 0 ||
      angle.atom2 <= 0 || angle.atom3 <= 0) {
    return -1;
  }
  angle.id -= 1;
  angle.type -= 1;
  angle.atom1 -= 1;
  angle.atom2 -= 1;
  angle.atom3 -= 1;
  if (streaming_mode_ && streaming_ctx_) {
    mpi::StreamAngle(*streaming_ctx_, angle);
    return 0;
  }
  snapshot_.angles.push_back(angle);
  return 0;
}

int LammpsFullReader::ParseDihedral(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::Dihedral dihedral;
  iss >> dihedral.id >> dihedral.type >> dihedral.atom1 >> dihedral.atom2 >>
      dihedral.atom3 >> dihedral.atom4;
  if (iss.fail()) {
    return -1;
  }
  if (dihedral.id <= 0 || dihedral.type <= 0 || dihedral.atom1 <= 0 ||
      dihedral.atom2 <= 0 || dihedral.atom3 <= 0 || dihedral.atom4 <= 0) {
    return -1;
  }
  dihedral.id -= 1;
  dihedral.type -= 1;
  dihedral.atom1 -= 1;
  dihedral.atom2 -= 1;
  dihedral.atom3 -= 1;
  dihedral.atom4 -= 1;
  if (streaming_mode_ && streaming_ctx_) {
    mpi::StreamDihedral(*streaming_ctx_, dihedral);
    return 0;
  }
  snapshot_.dihedrals.push_back(dihedral);
  return 0;
}

int LammpsFullReader::ParseImproper(std::string_view line) {
  auto content = StripComment(line);
  if (content.empty()) {
    return 0;
  }
  std::istringstream iss(content);
  ParsedSnapshot::Improper improper;
  iss >> improper.id >> improper.type >> improper.atom1 >> improper.atom2 >>
      improper.atom3 >> improper.atom4;
  if (iss.fail()) {
    return -1;
  }
  if (improper.id <= 0 || improper.type <= 0 || improper.atom1 <= 0 ||
      improper.atom2 <= 0 || improper.atom3 <= 0 || improper.atom4 <= 0) {
    return -1;
  }
  improper.id -= 1;
  improper.type -= 1;
  improper.atom1 -= 1;
  improper.atom2 -= 1;
  improper.atom3 -= 1;
  improper.atom4 -= 1;
  if (streaming_mode_ && streaming_ctx_) {
    mpi::StreamImproper(*streaming_ctx_, improper);
    return 0;
  }
  snapshot_.impropers.push_back(improper);
  return 0;
}

LammpsFullReader::Section LammpsFullReader::DetectSection(std::string_view line) const {
  auto trimmed = TrimCopy(line);
  if (trimmed.empty()) {
    return Section::None;
  }
  if (trimmed.rfind("Masses", 0) == 0) {
    return Section::Masses;
  }
  if (trimmed.rfind("Pair Coeffs", 0) == 0) {
    return Section::PairCoeffs;
  }
  if (trimmed.rfind("PairIJ Coeffs", 0) == 0) {
    return Section::PairCoeffs;
  }
  if (trimmed.rfind("Bond Coeffs", 0) == 0) {
    return Section::BondCoeffs;
  }
  if (trimmed.rfind("Angle Coeffs", 0) == 0) {
    return Section::AngleCoeffs;
  }
  if (trimmed.rfind("Dihedral Coeffs", 0) == 0) {
    return Section::DihedralCoeffs;
  }
  if (trimmed.rfind("Improper Coeffs", 0) == 0) {
    return Section::ImproperCoeffs;
  }
  if (trimmed.rfind("Atoms", 0) == 0) {
    return Section::Atoms;
  }
  if (trimmed.rfind("Velocities", 0) == 0) {
    return Section::Velocities;
  }
  if (trimmed.rfind("Bonds", 0) == 0) {
    return Section::Bonds;
  }
  if (trimmed.rfind("Angles", 0) == 0) {
    return Section::Angles;
  }
  if (trimmed.rfind("Dihedrals", 0) == 0) {
    return Section::Dihedrals;
  }
  if (trimmed.rfind("Impropers", 0) == 0) {
    return Section::Impropers;
  }
  return Section::None;
}

void LammpsFullReader::ResetSnapshot() {
  snapshot_ = ParsedSnapshot{};
  current_section_ = Section::None;
  expected_atoms_ = 0;
  expected_bonds_ = 0;
  expected_angles_ = 0;
  expected_dihedrals_ = 0;
  expected_impropers_ = 0;
  eam_potential_file_.clear();
  bond_adjacency_.clear();
  atom_image_flags_seen_ = false;
  atom_image_flags_missing_ = false;
  triclinic_box_detected_ = false;
  parsed_velocity_count_ = 0;
  atom_index_by_id_.clear();
}

void LammpsFullReader::PopulateDataManagers() {
  PopulateBox();
  PopulateStructureInfo();
#ifdef READER_ENABLE_MPI
  PopulateGlobalStructureInfo();
  CalculateAndPopulateTopologyParameters();
#endif
  PopulateStructure();
#ifdef READER_ENABLE_MPI
  if (auto full = std::dynamic_pointer_cast<FullStructureData>(
          data() ? data()->_structure_data : nullptr)) {
    auto device_data = DataManager::getInstance().getDeviceData();
    if (device_data) {
      // Call-chain contract for Task7A:
      // ConvertGlobalToPerAtomTopology -> CVFFMemoryScheduler::asyncMemoryH2D ->
      // CVFF per-atom topology ops in force/cvff.cpp.
      mpi::ConvertGlobalToPerAtomTopology(snapshot_, *full, *device_data);

      auto require_positive_per_atom = [](bool has_topology, int per_atom,
                                          const char* name) {
        if (has_topology && per_atom <= 0) {
          throw std::runtime_error(
              std::string("MPI topology conversion produced invalid ") + name +
              "_per_atom <= 0");
        }
      };
      require_positive_per_atom(!snapshot_.bonds.empty(), device_data->bond_per_atom,
                                "bond");
      require_positive_per_atom(!snapshot_.angles.empty(), device_data->angle_per_atom,
                                "angle");
      require_positive_per_atom(!snapshot_.dihedrals.empty(),
                                device_data->dihedral_per_atom, "dihedral");
      require_positive_per_atom(!snapshot_.impropers.empty(),
                                device_data->improper_per_atom, "improper");
    }
  }
#endif
  PopulateSpecialCounts();
  PopulateForceField();
}

void LammpsFullReader::PopulateBox() {
  auto storage = data();
  if (!storage || !storage->_box) {
    return;
  }
  auto box = storage->_box;
  box->_coord_min[0] = snapshot_.box.xlo;
  box->_coord_max[0] = snapshot_.box.xhi;
  box->_coord_min[1] = snapshot_.box.ylo;
  box->_coord_max[1] = snapshot_.box.yhi;
  box->_coord_min[2] = snapshot_.box.zlo;
  box->_coord_max[2] = snapshot_.box.zhi;
  //SetGlobalBox(box.get());
  bool periodic[3] = {true, true, true};   // TODO
#ifdef READER_ENABLE_MPI
  box->Setup(Box::BoxType::ORTHOGONAL, box->_coord_min, box->_coord_max, periodic);
  Box* global_box_ = &GET_RBMD_PARALLEL->_global_structure_info.global_box;
  global_box_->Setup(Box::BoxType::ORTHOGONAL, box->_coord_min, box->_coord_max, periodic);
#else
  box->Setup(Box::BoxType::ORTHOGONAL, box->_coord_min, box->_coord_max, periodic);
#endif

}

void LammpsFullReader::PopulateStructure() {
  auto storage = data();
  if (!storage || !storage->_structure_data) {
    throw std::runtime_error("LAMMPS reader missing structure storage");
  }

  auto adapter = CreateStructureAdapter(storage->_structure_data);

  if (!adapter) {
    throw std::runtime_error("Unsupported structure data type for LAMMPS reader");
  }

  adapter->Populate(snapshot_);
}

void LammpsFullReader::PopulateStructureInfo() {
  auto storage = data();
  if (!storage || !storage->_structure_info_data) {
    return;
  }

  auto info = storage->_structure_info_data;
  auto structure_data = storage->_structure_data;
  const bool has_full_topology =
      structure_data &&
      static_cast<bool>(std::dynamic_pointer_cast<FullStructureData>(structure_data));

  // 使用持久化内存分配，而不是临时的 owner.allocateScalar
  // 这些指针会在 StructureInfoData 的析构函数中被正确释放

  // 先释放旧的内存（如果存在）
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_atoms));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_bonds));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_angles));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_dihedrals));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_impropers));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_atoms_type));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_bonds_type));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_angles_type));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_dihedrals_type));
  CHECK_RUNTIME(FREE_PINNED_HOST(info->_num_impropers_type));

  // 分配新的持久化内存
  CHECK_RUNTIME(MALLOCHOST(&info->_num_atoms, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_bonds, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_angles, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_dihedrals, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_impropers, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_atoms_type, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_bonds_type, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_angles_type, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_dihedrals_type, sizeof(rbmd::Id)));
  CHECK_RUNTIME(MALLOCHOST(&info->_num_impropers_type, sizeof(rbmd::Id)));

  // 填充数据
  *info->_num_atoms = static_cast<rbmd::Id>(snapshot_.atoms.size());
  *info->_num_bonds = static_cast<rbmd::Id>(has_full_topology ? snapshot_.bonds.size() : 0);
  *info->_num_angles = static_cast<rbmd::Id>(has_full_topology ? snapshot_.angles.size() : 0);
  *info->_num_dihedrals = static_cast<rbmd::Id>(has_full_topology ? snapshot_.dihedrals.size() : 0);
  *info->_num_impropers = static_cast<rbmd::Id>(has_full_topology ? snapshot_.impropers.size() : 0);
  *info->_num_atoms_type = snapshot_.topology.atom_types;
  *info->_num_bonds_type = has_full_topology ? snapshot_.topology.bond_types : 0;
  *info->_num_angles_type = has_full_topology ? snapshot_.topology.angle_types : 0;
  *info->_num_dihedrals_type = has_full_topology ? snapshot_.topology.dihedral_types : 0;
  *info->_num_impropers_type = has_full_topology ? snapshot_.topology.improper_types : 0;
}

#ifdef READER_ENABLE_MPI
void LammpsFullReader::PopulateGlobalStructureInfo() {
  const int rank = mpi::IsEnabled() ? mpi::Rank() : 0;
  if (rank != 0) {
    return;
  }

  auto storage = data();
  if (!storage || !storage->_structure_data) {
    return;
  }

  const bool has_full_topology =
      storage->_structure_data &&
      static_cast<bool>(std::dynamic_pointer_cast<FullStructureData>(storage->_structure_data));

  // 使用头部读取到的全局 expected_*，而不是 MPI 分发后的局部 snapshot 大小。
  GlobalStructureInfo& global_info = GET_RBMD_PARALLEL->_global_structure_info;

  global_info.total_atoms = static_cast<rbmd::Id>(expected_atoms_);
  global_info.total_bonds = static_cast<rbmd::Id>(has_full_topology ? expected_bonds_ : 0);
  global_info.total_angles = static_cast<rbmd::Id>(has_full_topology ? expected_angles_ : 0);
  global_info.total_dihedrals = static_cast<rbmd::Id>(has_full_topology ? expected_dihedrals_ : 0);
  global_info.total_impropers = static_cast<rbmd::Id>(has_full_topology ? expected_impropers_ : 0);

  global_info.num_atom_types = snapshot_.topology.atom_types;
  global_info.num_bond_types = has_full_topology ? snapshot_.topology.bond_types : 0;
  global_info.num_angle_types = has_full_topology ? snapshot_.topology.angle_types : 0;
  global_info.num_dihedral_types = has_full_topology ? snapshot_.topology.dihedral_types : 0;
  
  // global_box 已经在 PopulateBox() 中填充
}
#endif

void LammpsFullReader::PopulateSpecialCounts() {
  auto storage = data();
  if (!storage || !storage->_structure_data) {
    bond_adjacency_.clear();
    return;
  }

  auto full = std::dynamic_pointer_cast<FullStructureData>(storage->_structure_data);
  if (!full) {
    bond_adjacency_.clear();
    return;
  }

  auto reset_special_data = [](FullStructureData& data) {
    if (data._h_special_weights) {
      std::free(data._h_special_weights);
      data._h_special_weights = nullptr;
    }
    if (data._h_special_ids) {
      std::free(data._h_special_ids);
      data._h_special_ids = nullptr;
    }
    if (data._h_special_offsets) {
      std::free(data._h_special_offsets);
      data._h_special_offsets = nullptr;
    }
    if (data._h_special_offset_count) {
      std::free(data._h_special_offset_count);
      data._h_special_offset_count = nullptr;
    }
    if (data._h_atoms_vec_gro) {
      std::free(data._h_atoms_vec_gro);
      data._h_atoms_vec_gro = nullptr;
    }
    if (data._h_count_vector) {
      std::free(data._h_count_vector);
      data._h_count_vector = nullptr;
    }
    if (data._h_atoms_offset) {
      std::free(data._h_atoms_offset);
      data._h_atoms_offset = nullptr;
    }
    if (data._h_nspecial) {
      std::free(data._h_nspecial);
      data._h_nspecial = nullptr;
    }
    if (data._h_special) {
      std::free(data._h_special);
      data._h_special = nullptr;
    }
    data._num_special_weights = 0;
    data._num_special_ids = 0;
    data._num_special_offsets = 0;
    data._num_special_offset_count = 0;
    data._num_atoms_vec_gro = 0;
    data._num_count_vector = 0;
    data._num_atoms_offset = 0;
    data._h_maxspecial = 0;
    data._h_nmax_special = 0;
  };

#ifdef READER_ENABLE_MPI
  if (mpi::IsEnabled()) {
    // MPI 模式：
    // - ConvertGlobalToPerAtomTopology() 已生成 per-atom topology
    // - 直接把 per-atom 记录展开成 local_bonds/local_angles/local_dihedrals
    //   交由 SpecialBondBuilder 按 LAMMPS(newton off) 语义构建 special。
    auto device_data = DataManager::getInstance().getDeviceData();
    if (!device_data) {
      reset_special_data(*full);
      bond_adjacency_.clear();
      return;
    }

    std::vector<rbmd::Id> local_atom_ids;
    local_atom_ids.reserve(snapshot_.atoms.size());
    for (const auto& atom : snapshot_.atoms) {
      local_atom_ids.push_back(atom.id);
    }

    std::vector<rbmd::Id> ghost_atom_ids;

    std::unordered_set<rbmd::Id> local_atom_set(local_atom_ids.begin(),
                                                local_atom_ids.end());

    std::vector<std::tuple<rbmd::Id, rbmd::Id>> local_bonds;
    local_bonds.reserve(snapshot_.bonds.size() * 2);
    for (const auto& bond : snapshot_.bonds) {
      if (local_atom_set.count(bond.atom1) > 0) {
        local_bonds.emplace_back(bond.atom1, bond.atom2);
      }
      if (local_atom_set.count(bond.atom2) > 0) {
        local_bonds.emplace_back(bond.atom2, bond.atom1);
      }
    }

    std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id>> local_angles;
    const int angle_per_atom = device_data->angle_per_atom;
    if (full->_h_num_angle && full->_h_angle_type_per_atom &&
        full->_h_angle_atom1_per_atom && full->_h_angle_atom2_per_atom &&
        full->_h_angle_atom3_per_atom && angle_per_atom > 0) {
      for (std::size_t i = 0; i < snapshot_.atoms.size(); ++i) {
        const int num_angles = full->_h_num_angle[i];
        for (int j = 0; j < num_angles; ++j) {
          const std::size_t offset =
              i * static_cast<std::size_t>(angle_per_atom) + j;
          local_angles.emplace_back(
              full->_h_angle_type_per_atom[offset],
              full->_h_angle_atom1_per_atom[offset],
              full->_h_angle_atom2_per_atom[offset],
              full->_h_angle_atom3_per_atom[offset]);
        }
      }
    }

    std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id, rbmd::Id>>
        local_dihedrals;
    const int dihedral_per_atom = device_data->dihedral_per_atom;
    if (full->_h_num_dihedral && full->_h_dihedral_type_per_atom &&
        full->_h_dihedral_atom1_per_atom && full->_h_dihedral_atom2_per_atom &&
        full->_h_dihedral_atom3_per_atom && full->_h_dihedral_atom4_per_atom &&
        dihedral_per_atom > 0) {
      for (std::size_t i = 0; i < snapshot_.atoms.size(); ++i) {
        const int num_dihedrals = full->_h_num_dihedral[i];
        for (int j = 0; j < num_dihedrals; ++j) {
          const std::size_t offset =
              i * static_cast<std::size_t>(dihedral_per_atom) + j;
          local_dihedrals.emplace_back(
              full->_h_dihedral_type_per_atom[offset],
              full->_h_dihedral_atom1_per_atom[offset],
              full->_h_dihedral_atom2_per_atom[offset],
              full->_h_dihedral_atom3_per_atom[offset],
              full->_h_dihedral_atom4_per_atom[offset]);
        }
      }
    }

    int my_rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    reader::mpi::SpecialBondBuilder builder;
    builder.Build(
        std::move(local_atom_ids),
        std::move(ghost_atom_ids),
        std::move(local_bonds),
        local_angles,
        local_dihedrals,
        my_rank,
        num_ranks);

    bond_adjacency_.clear();
    return;
  }
#endif

  const auto atom_count = snapshot_.atoms.size();
  if (atom_count == 0) {
    reset_special_data(*full);
    bond_adjacency_.clear();
    return;
  }

  // 参考 ref/atomic_reader.cpp 的 special 处理：使用 Bonds/Angles/Dihedrals
  // 构建 1-2/1-3/1-4 的 special 列表，并通过“有序对”去重以保证优先级：
  // 1-2 覆盖 1-3，1-3 覆盖 1-4。
  std::vector<std::vector<rbmd::Id>> atom_neighbors(atom_count);
  std::vector<std::vector<rbmd::Real>> atom_weights(atom_count);
  std::vector<std::vector<int>> atom_bands(atom_count);
  std::unordered_map<rbmd::Id, std::size_t> local_index_by_id;
  local_index_by_id.reserve(atom_count);
  for (std::size_t i = 0; i < atom_count; ++i) {
    local_index_by_id.emplace(snapshot_.atoms[i].id, i);
  }
  std::unordered_set<std::pair<rbmd::Id, rbmd::Id>, PairHash> excluded_pairs;
  excluded_pairs.reserve(snapshot_.bonds.size() +
                         snapshot_.angles.size() +
                         snapshot_.dihedrals.size());

  const auto add_pair = [&](rbmd::Id a, rbmd::Id b, rbmd::Real weight,
                            int band) {
    if (a < 0 || b < 0 || a == b) {
      return;
    }
    const auto ai = local_index_by_id.find(a);
    const auto bi = local_index_by_id.find(b);
    if (ai == local_index_by_id.end() || bi == local_index_by_id.end()) {
      return;
    }
    auto key = OrderedPair(a, b);
    if (!excluded_pairs.insert(key).second) {
      return;
    }
    atom_neighbors[ai->second].push_back(b);
    atom_weights[ai->second].push_back(weight);
    atom_bands[ai->second].push_back(band);
    atom_neighbors[bi->second].push_back(a);
    atom_weights[bi->second].push_back(weight);
    atom_bands[bi->second].push_back(band);
  };

  for (const auto& bond : snapshot_.bonds) {
    add_pair(bond.atom1, bond.atom2, special_bond_weights_[0], 0);
  }
  for (const auto& angle : snapshot_.angles) {
    add_pair(angle.atom1, angle.atom3, special_bond_weights_[1], 1);
  }
  for (const auto& dihedral : snapshot_.dihedrals) {
    add_pair(dihedral.atom1, dihedral.atom4, special_bond_weights_[2], 2);
  }

  std::vector<rbmd::Id> offset_counts(atom_count, 0);
  std::vector<rbmd::Id> offsets(atom_count + 1, 0);
  rbmd::Id total_pairs = 0;
  for (std::size_t i = 0; i < atom_count; ++i) {
    offset_counts[i] = static_cast<rbmd::Id>(atom_neighbors[i].size());
    offsets[i + 1] = offsets[i] + offset_counts[i];
  }
  total_pairs = offsets.back();

  std::vector<rbmd::Real> special_weights(static_cast<std::size_t>(total_pairs));
  std::vector<rbmd::Id> special_ids(static_cast<std::size_t>(total_pairs));
  int maxspecial = 0;
  rbmd::Id cursor = 0;
  for (std::size_t i = 0; i < atom_count; ++i) {
    maxspecial = std::max(maxspecial, static_cast<int>(atom_neighbors[i].size()));
    for (std::size_t j = 0; j < atom_neighbors[i].size(); ++j) {
      special_ids[static_cast<std::size_t>(cursor)] = atom_neighbors[i][j];
      special_weights[static_cast<std::size_t>(cursor)] = atom_weights[i][j];
      ++cursor;
    }
  }

  reset_special_data(*full);

  if (special_weights.empty()) {
    bond_adjacency_.clear();
    return;
  }

  full->_num_special_weights = static_cast<rbmd::Id>(special_weights.size());
  full->_num_special_ids = static_cast<rbmd::Id>(special_ids.size());
  full->_num_special_offset_count = static_cast<rbmd::Id>(offset_counts.size());
  full->_num_special_offsets = static_cast<rbmd::Id>(offsets.size());

  full->_h_special_weights =
      detail::AllocateStructureArray<rbmd::Real>(special_weights.size());
  full->_h_special_ids =
      detail::AllocateStructureArray<rbmd::Id>(special_ids.size());
  full->_h_special_offset_count =
      detail::AllocateStructureArray<rbmd::Id>(offset_counts.size());
  full->_h_special_offsets =
      detail::AllocateStructureArray<rbmd::Id>(offsets.size());
  if (maxspecial > 0) {
    const std::size_t nspecial_size = atom_count * static_cast<std::size_t>(3);
    const std::size_t special_size =
        atom_count * static_cast<std::size_t>(maxspecial);
    full->_h_nspecial = detail::AllocateStructureArray<int>(nspecial_size);
    full->_h_special = detail::AllocateStructureArray<rbmd::Id>(special_size);
    full->_h_maxspecial = maxspecial;
    full->_h_nmax_special = static_cast<rbmd::Id>(atom_count);
    std::memset(full->_h_nspecial, 0, nspecial_size * sizeof(int));
    std::memset(full->_h_special, 0, special_size * sizeof(rbmd::Id));
    for (std::size_t i = 0; i < atom_count; ++i) {
      int count12 = 0;
      int count13 = 0;
      int count14 = 0;
      for (std::size_t j = 0; j < atom_neighbors[i].size(); ++j) {
        full->_h_special[i * static_cast<std::size_t>(maxspecial) + j] =
            atom_neighbors[i][j];
        const int band = atom_bands[i][j];
        if (band == 0) {
          ++count12;
        } else if (band == 1) {
          ++count13;
        } else {
          ++count14;
        }
      }
      full->_h_nspecial[i * 3 + 0] = count12;
      full->_h_nspecial[i * 3 + 1] = count12 + count13;
      full->_h_nspecial[i * 3 + 2] = count12 + count13 + count14;
    }
  }

  std::memcpy(full->_h_special_weights,
              special_weights.data(),
              special_weights.size() * sizeof(rbmd::Real));
  std::memcpy(full->_h_special_ids,
              special_ids.data(),
              special_ids.size() * sizeof(rbmd::Id));
  std::memcpy(full->_h_special_offset_count,
              offset_counts.data(),
              offset_counts.size() * sizeof(rbmd::Id));
  std::memcpy(full->_h_special_offsets,
              offsets.data(),
              offsets.size() * sizeof(rbmd::Id));

  bond_adjacency_.clear();
}

void LammpsFullReader::PopulateForceField() {
  auto storage = data();
  if (!storage || !storage->_force_field_data) {
    return;
  }

  ValidateRequiredCoeffSectionsOrThrow();

  if (IsEamForceField()) {
    ResolveEamPotentialFile();
  }

  auto& owner = dataOwner();
  const auto atom_type_count = static_cast<std::size_t>(
      std::max<rbmd::Id>(snapshot_.topology.atom_types,
                         static_cast<rbmd::Id>(snapshot_.masses.size())));
  if (auto lj = std::dynamic_pointer_cast<LJForceFieldData>(
          storage->_force_field_data)) {
    if (atom_type_count == 0) {
      lj->_h_mass = nullptr;
      lj->_h_eps = nullptr;
      lj->_h_sigma = nullptr;
    } else {
      auto* masses = AllocateTrackedArray<rbmd::Real>(owner, atom_type_count);
      auto* eps = AllocateTrackedArray<rbmd::Real>(owner, atom_type_count);
      auto* sigma = AllocateTrackedArray<rbmd::Real>(owner, atom_type_count);
      FillZero(masses, atom_type_count);
      FillZero(eps, atom_type_count);
      FillZero(sigma, atom_type_count);

      for (const auto& entry : snapshot_.masses) {
        if (entry.first < 0) {
          continue;
        }
        const auto index = static_cast<std::size_t>(entry.first);
        if (index < atom_type_count) {
          masses[index] = entry.second;
        }
      }
      for (const auto& entry : snapshot_.pair_coeffs) {
        if (entry.first < 0) {
          continue;
        }
        const auto index = static_cast<std::size_t>(entry.first);
        if (index < atom_type_count) {
          eps[index] = entry.second.epsilon;
          sigma[index] = entry.second.sigma;
        }
      }

      lj->_h_mass = masses;
      lj->_h_eps = eps;
      lj->_h_sigma = sigma;
    }
    return;
  }

  if (auto cvff = std::dynamic_pointer_cast<CVFFForceFieldData>(
          storage->_force_field_data)) {
    if (atom_type_count == 0) {
      cvff->_h_mass = nullptr;
      cvff->_h_eps = nullptr;
      cvff->_h_sigma = nullptr;
    } else {
      cvff->_h_mass = AllocateTrackedArray<rbmd::Real>(owner, atom_type_count);
      cvff->_h_eps = AllocateTrackedArray<rbmd::Real>(owner, atom_type_count);
      cvff->_h_sigma = AllocateTrackedArray<rbmd::Real>(owner, atom_type_count);
      FillZero(cvff->_h_mass, atom_type_count);
      FillZero(cvff->_h_eps, atom_type_count);
      FillZero(cvff->_h_sigma, atom_type_count);

      for (const auto& entry : snapshot_.masses) {
        if (entry.first < 0) {
          continue;
        }
        const auto index = static_cast<std::size_t>(entry.first);
        if (index < atom_type_count) {
          cvff->_h_mass[index] = entry.second;
        }
      }
      for (const auto& entry : snapshot_.pair_coeffs) {
        if (entry.first < 0) {
          continue;
        }
        const auto index = static_cast<std::size_t>(entry.first);
        if (index < atom_type_count) {
          cvff->_h_eps[index] = entry.second.epsilon;
          cvff->_h_sigma[index] = entry.second.sigma;
        }
      }
    }

    const auto bond_type_count = static_cast<std::size_t>(
        std::max<rbmd::Id>(snapshot_.topology.bond_types,
                           static_cast<rbmd::Id>(snapshot_.bond_coeffs.size())));
    cvff->_h_bond_coeffs_k =
        AllocateTrackedArray<rbmd::Real>(owner, bond_type_count);
    cvff->_h_bond_coeffs_equilibrium =
        AllocateTrackedArray<rbmd::Real>(owner, bond_type_count);
    FillZero(cvff->_h_bond_coeffs_k, bond_type_count);
    FillZero(cvff->_h_bond_coeffs_equilibrium, bond_type_count);
    for (const auto& entry : snapshot_.bond_coeffs) {
      if (entry.first < 0) {
        continue;
      }
      const auto index = static_cast<std::size_t>(entry.first);
      if (index < bond_type_count) {
        cvff->_h_bond_coeffs_k[index] = entry.second.k;
        cvff->_h_bond_coeffs_equilibrium[index] = entry.second.equilibrium;
      }
    }
    ValidateBondCoeffArrayOrThrow("after_bond_fill", "bond_equilibrium",
                                  cvff->_h_bond_coeffs_equilibrium,
                                  bond_type_count);

    const auto angle_type_count = static_cast<std::size_t>(
        std::max<rbmd::Id>(snapshot_.topology.angle_types,
                           static_cast<rbmd::Id>(snapshot_.angle_coeffs.size())));
    cvff->_h_angle_coeffs_k =
        AllocateTrackedArray<rbmd::Real>(owner, angle_type_count);
    cvff->_h_angle_coeffs_equilibrium =
        AllocateTrackedArray<rbmd::Real>(owner, angle_type_count);
    FillZero(cvff->_h_angle_coeffs_k, angle_type_count);
    FillZero(cvff->_h_angle_coeffs_equilibrium, angle_type_count);
    for (const auto& entry : snapshot_.angle_coeffs) {
      if (entry.first < 0) {
        continue;
      }
      const auto index = static_cast<std::size_t>(entry.first);
      if (index < angle_type_count) {
        cvff->_h_angle_coeffs_k[index] = entry.second.k;
        cvff->_h_angle_coeffs_equilibrium[index] = entry.second.equilibrium;
      }
    }

    const auto dihedral_type_count = static_cast<std::size_t>(
        std::max<rbmd::Id>(snapshot_.topology.dihedral_types,
                           static_cast<rbmd::Id>(snapshot_.dihedral_coeffs.size())));
    const auto improper_type_count = static_cast<std::size_t>(
        std::max<rbmd::Id>(snapshot_.topology.improper_types,
                           static_cast<rbmd::Id>(snapshot_.improper_coeffs.size())));

    std::string dihedral_type = "harmonic";
    std::string improper_type = "harmonic";
    if (auto config = DataManager::getInstance().getConfigData()) {
      try {
        dihedral_type = config->Get<std::string>("dihedral_type",
                                                 "hyper_parameters",
                                                 "force_field");
      } catch (const std::exception&) {
      }
      try {
        improper_type = config->Get<std::string>("improper_type",
                                                 "hyper_parameters",
                                                 "force_field");
      } catch (const std::exception&) {
      }
    }
    auto to_lower = [](std::string value) {
      std::transform(value.begin(), value.end(), value.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      return value;
    };
    dihedral_type = to_lower(dihedral_type);
    improper_type = to_lower(improper_type);

    cvff->_h_dihedral_coeffs_k = nullptr;
    cvff->_h_dihedral_coeffs_sign = nullptr;
    cvff->_h_dihedral_coeffs_multiplicity = nullptr;
    cvff->_h_dihedral_coeffs_k1 = nullptr;
    cvff->_h_dihedral_coeffs_k2 = nullptr;
    cvff->_h_dihedral_coeffs_k3 = nullptr;
    cvff->_h_dihedral_coeffs_k4 = nullptr;

    if (dihedral_type_count > 0) {
      if (dihedral_type == "opls") {
        cvff->_h_dihedral_coeffs_k1 =
            AllocateTrackedArray<rbmd::Real>(owner, dihedral_type_count);
        cvff->_h_dihedral_coeffs_k2 =
            AllocateTrackedArray<rbmd::Real>(owner, dihedral_type_count);
        cvff->_h_dihedral_coeffs_k3 =
            AllocateTrackedArray<rbmd::Real>(owner, dihedral_type_count);
        cvff->_h_dihedral_coeffs_k4 =
            AllocateTrackedArray<rbmd::Real>(owner, dihedral_type_count);
        FillZero(cvff->_h_dihedral_coeffs_k1, dihedral_type_count);
        FillZero(cvff->_h_dihedral_coeffs_k2, dihedral_type_count);
        FillZero(cvff->_h_dihedral_coeffs_k3, dihedral_type_count);
        FillZero(cvff->_h_dihedral_coeffs_k4, dihedral_type_count);
        for (const auto& entry : snapshot_.dihedral_coeffs) {
          if (entry.first < 0) {
            continue;
          }
          const auto index = static_cast<std::size_t>(entry.first);
          if (index < dihedral_type_count) {
            cvff->_h_dihedral_coeffs_k1[index] = entry.second.k1;
            cvff->_h_dihedral_coeffs_k2[index] = entry.second.k2;
            cvff->_h_dihedral_coeffs_k3[index] = entry.second.k3;
            cvff->_h_dihedral_coeffs_k4[index] = entry.second.k4;
          }
        }
      } else {
        cvff->_h_dihedral_coeffs_k =
            AllocateTrackedArray<rbmd::Real>(owner, dihedral_type_count);
        cvff->_h_dihedral_coeffs_sign =
            AllocateTrackedArray<rbmd::Id>(owner, dihedral_type_count);
        cvff->_h_dihedral_coeffs_multiplicity =
            AllocateTrackedArray<rbmd::Id>(owner, dihedral_type_count);
        FillZero(cvff->_h_dihedral_coeffs_k, dihedral_type_count);
        FillZero(cvff->_h_dihedral_coeffs_sign, dihedral_type_count);
        FillZero(cvff->_h_dihedral_coeffs_multiplicity, dihedral_type_count);
        for (const auto& entry : snapshot_.dihedral_coeffs) {
          if (entry.first < 0) {
            continue;
          }
          const auto index = static_cast<std::size_t>(entry.first);
          if (index < dihedral_type_count) {
            cvff->_h_dihedral_coeffs_k[index] = entry.second.k;
            cvff->_h_dihedral_coeffs_sign[index] = entry.second.sign;
            cvff->_h_dihedral_coeffs_multiplicity[index] =
                entry.second.multiplicity;
          }
        }
      }
    }

    cvff->_h_improper_coeffs_k = nullptr;
    cvff->_h_improper_coeffs_degree = nullptr;
    cvff->_h_improper_coeffs_d = nullptr;
    cvff->_h_improper_coeffs_n = nullptr;

    if (improper_type_count > 0) {
      cvff->_h_improper_coeffs_k =
          AllocateTrackedArray<rbmd::Real>(owner, improper_type_count);
      FillZero(cvff->_h_improper_coeffs_k, improper_type_count);

      if (improper_type == "cvff") {
        cvff->_h_improper_coeffs_d =
            AllocateTrackedArray<rbmd::Id>(owner, improper_type_count);
        cvff->_h_improper_coeffs_n =
            AllocateTrackedArray<rbmd::Id>(owner, improper_type_count);
        FillZero(cvff->_h_improper_coeffs_d, improper_type_count);
        FillZero(cvff->_h_improper_coeffs_n, improper_type_count);
      } else {
        cvff->_h_improper_coeffs_degree =
            AllocateTrackedArray<rbmd::Real>(owner, improper_type_count);
        FillZero(cvff->_h_improper_coeffs_degree, improper_type_count);
      }

      for (const auto& entry : snapshot_.improper_coeffs) {
        if (entry.first < 0) {
          continue;
        }
        const auto index = static_cast<std::size_t>(entry.first);
        if (index >= improper_type_count) {
          continue;
        }
        if (cvff->_h_improper_coeffs_k) {
          cvff->_h_improper_coeffs_k[index] = entry.second.k;
        }
        if (cvff->_h_improper_coeffs_degree) {
          cvff->_h_improper_coeffs_degree[index] = entry.second.degree;
        }
        if (cvff->_h_improper_coeffs_d) {
          cvff->_h_improper_coeffs_d[index] = entry.second.d;
        }
        if (cvff->_h_improper_coeffs_n) {
          cvff->_h_improper_coeffs_n[index] = entry.second.n;
        }
      }
    }
    ValidateBondCoeffArrayOrThrow("after_forcefield_fill", "bond_equilibrium",
                                  cvff->_h_bond_coeffs_equilibrium,
                                  bond_type_count);
  }
}

void LammpsFullReader::ValidateRequiredCoeffSectionsOrThrow() const {
  if (!IsEamForceField()) {
    ValidateCoeffCoverageOrThrow("Pair Coeffs", snapshot_.topology.atom_types,
                                 snapshot_.pair_coeffs);
  }

  ValidateCoeffCoverageOrThrow("Bond Coeffs", snapshot_.topology.bond_types,
                               snapshot_.bond_coeffs);
  ValidateCoeffCoverageOrThrow("Angle Coeffs", snapshot_.topology.angle_types,
                               snapshot_.angle_coeffs);
}

bool LammpsFullReader::IsEamForceField() const {
  auto config = DataManager::getInstance().getConfigData();
  if (!config) {
    return false;
  }

  try {
    auto force_type = config->Get<std::string>(
        "type", "hyper_parameters", "force_field");
    std::transform(force_type.begin(), force_type.end(), force_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return force_type == "eam";
  } catch (const std::exception&) {
    return false;
  }
}

void LammpsFullReader::ResolveEamPotentialFile() {
  auto config = DataManager::getInstance().getConfigData();
  if (!config) {
    return;
  }

  std::string candidate = eam_potential_file_;
  if (candidate.empty()) {
    try {
      candidate = config->Get<std::string>(
          "potential_file", "hyper_parameters", "force_field");
    } catch (const std::exception&) {
      return;
    }
  }

  if (candidate.empty()) {
    return;
  }

  fs::path candidate_path(candidate);
  std::string resolved = candidate;

  auto normalize = [](const fs::path& path) -> std::string {
    try {
      if (fs::exists(path)) {
        return fs::canonical(path).string();
      }
    } catch (const std::exception&) {
    }
    return path.string();
  };

  if (candidate_path.is_absolute()) {
    resolved = normalize(candidate_path);
  } else {
    std::vector<fs::path> search_roots;
    const fs::path data_dir = fs::path(filePath()).parent_path();
    const fs::path json_dir(config->GetConfigDir());

    if (!eam_potential_file_.empty()) {
      if (!data_dir.empty()) {
        search_roots.push_back(data_dir);
      }
      if (!json_dir.empty()) {
        search_roots.push_back(json_dir);
      }
    } else {
      if (!json_dir.empty()) {
        search_roots.push_back(json_dir);
      }
      if (!data_dir.empty()) {
        search_roots.push_back(data_dir);
      }
    }
    search_roots.push_back(fs::current_path());

    for (const auto& root : search_roots) {
      fs::path attempt = root / candidate_path;
      if (fs::exists(attempt)) {
        resolved = normalize(attempt);
        break;
      }
    }

    if (resolved == candidate && !search_roots.empty()) {
      resolved = (search_roots.front() / candidate_path).string();
    }
  }

  auto& hyper = config->GetJsonNode("hyper_parameters");
  hyper["force_field"]["potential_file"] = resolved;
}

void LammpsFullReader::LoadSpecialBondWeights() {
  special_bond_weights_ = {rbmd::Real{1.0}, rbmd::Real{1.0}, rbmd::Real{1.0}};
  try {
    if (auto& config_ptr = DataManager::getInstance().getConfigData()) {
      auto values = config_ptr->GetArray<rbmd::Real>(
          "special_bonds", "hyper_parameters", "extend");
      if (!values.empty()) {
        special_bond_weights_[0] = values[0];
        if (values.size() > 1) {
          special_bond_weights_[1] = values[1];
        }
        if (values.size() > 2) {
          special_bond_weights_[2] = values[2];
        }
      }
    }
  } catch (const std::exception&) {
    // keep defaults when configuration is unavailable
  }
}

LammpsFullReader::AtomStyle LammpsFullReader::DetectAtomStyle() const {
  auto storage = data();
  if (!storage || !storage->_structure_data) {
    return AtomStyle::Full;
  }
  const auto& structure = storage->_structure_data;
  if (std::dynamic_pointer_cast<FullStructureData>(structure)) {
    return AtomStyle::Full;
  }
  if (std::dynamic_pointer_cast<ChargeStructureData>(structure)) {
    return AtomStyle::Charge;
  }
  if (std::dynamic_pointer_cast<AtomsStructureData>(structure)) {
    return AtomStyle::Atomic;
  }
  return AtomStyle::Full;
}

std::string LammpsFullReader::Trim(std::string_view value) {
  return TrimCopy(value);
}

bool LammpsFullReader::IsComment(std::string_view value) {
  auto trimmed = TrimCopy(value);
  return trimmed.empty() || trimmed.front() == '#';
}

#ifdef READER_ENABLE_MPI
void LammpsFullReader::CalculateAndPopulateTopologyParameters() {
  auto storage = data();
  if (!storage) {
    return;
  }

  auto device_data = DataManager::getInstance().getDeviceData();
  if (!device_data) {
    return;
  }

  const bool has_full_topology =
      storage->_structure_data &&
      static_cast<bool>(
          std::dynamic_pointer_cast<FullStructureData>(storage->_structure_data));

  const int nprocs = mpi::IsEnabled() ? mpi::Size() : 1;
  constexpr double LB_FACTOR = 1.5;
  if (nprocs == 1) {
    device_data->nmax = static_cast<rbmd::Id>(expected_atoms_);
  } else {
    device_data->nmax =
        static_cast<rbmd::Id>(LB_FACTOR * expected_atoms_ / nprocs);
  }

  if (!has_full_topology) {
    device_data->bond_per_atom = 0;
    device_data->angle_per_atom = 0;
    device_data->dihedral_per_atom = 0;
    device_data->improper_per_atom = 0;
    device_data->maxspecial = 1;
    return;
  }

  const std::size_t num_atoms = snapshot_.atoms.size();
  std::vector<int> bond_count(num_atoms, 0);
  std::vector<int> angle_count(num_atoms, 0);
  std::vector<int> dihedral_count(num_atoms, 0);
  std::vector<int> improper_count(num_atoms, 0);

  std::unordered_map<rbmd::Id, std::size_t> gid_to_local;
  gid_to_local.reserve(num_atoms);
  for (std::size_t local = 0; local < num_atoms; ++local) {
    gid_to_local.emplace(snapshot_.atoms[local].id, local);
  }

  auto bump_if_local = [&](rbmd::Id gid, std::vector<int>& counter) {
    auto it = gid_to_local.find(gid);
    if (it != gid_to_local.end()) {
      counter[it->second] += 1;
    }
  };

  for (const auto& bond : snapshot_.bonds) {
    bump_if_local(bond.atom1, bond_count);
    bump_if_local(bond.atom2, bond_count);
  }
  for (const auto& angle : snapshot_.angles) {
    bump_if_local(angle.atom1, angle_count);
    bump_if_local(angle.atom2, angle_count);
    bump_if_local(angle.atom3, angle_count);
  }
  for (const auto& dihedral : snapshot_.dihedrals) {
    bump_if_local(dihedral.atom1, dihedral_count);
    bump_if_local(dihedral.atom2, dihedral_count);
    bump_if_local(dihedral.atom3, dihedral_count);
    bump_if_local(dihedral.atom4, dihedral_count);
  }
  for (const auto& improper : snapshot_.impropers) {
    bump_if_local(improper.atom1, improper_count);
    bump_if_local(improper.atom2, improper_count);
    bump_if_local(improper.atom3, improper_count);
    bump_if_local(improper.atom4, improper_count);
  }

  int local_max_bonds =
      bond_count.empty() ? 0 : *std::max_element(bond_count.begin(), bond_count.end());
  int local_max_angles =
      angle_count.empty() ? 0 : *std::max_element(angle_count.begin(), angle_count.end());
  int local_max_dihedrals = dihedral_count.empty()
                                ? 0
                                : *std::max_element(dihedral_count.begin(), dihedral_count.end());
  int local_max_impropers = improper_count.empty()
                                ? 0
                                : *std::max_element(improper_count.begin(), improper_count.end());

  int global_max_bonds = 0;
  int global_max_angles = 0;
  int global_max_dihedrals = 0;
  int global_max_impropers = 0;
  if (mpi::IsEnabled()) {
    MPI_Allreduce(&local_max_bonds, &global_max_bonds, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_angles, &global_max_angles, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_dihedrals, &global_max_dihedrals, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_impropers, &global_max_impropers, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
  } else {
    global_max_bonds = local_max_bonds;
    global_max_angles = local_max_angles;
    global_max_dihedrals = local_max_dihedrals;
    global_max_impropers = local_max_impropers;
  }

  const int extra_margin = 2;
  device_data->bond_per_atom = global_max_bonds + extra_margin;
  device_data->angle_per_atom = global_max_angles + extra_margin;
  device_data->dihedral_per_atom = global_max_dihedrals + extra_margin;
  device_data->improper_per_atom = global_max_impropers + extra_margin;

  const int estimated_maxspecial = global_max_bonds * 3;
  device_data->maxspecial = MAX(estimated_maxspecial, 12);
}
#endif

}  // namespace reader
