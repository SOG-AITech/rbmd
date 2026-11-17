#include "common/revision_timing.h"
#include "common/revision_diagnostics.h"
#include "common/reference_force.h"
#include "common/mpi_root_guard.hpp"
#include "common/rbmd_define.h"
#include "common/thermo_stats.hpp"
#include "common/timing_statistics.hpp"
#include "common/unit_factor.h"
#include <thrust/host_vector.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <unordered_map>

namespace rbmd { namespace revision {
namespace {
std::ofstream forces, observables;
std::ofstream force_errors;
struct LocalReference {
  ReferenceForce reference{};
  OnlineForceMoments moments;
  std::array<double, 3> frozen_position{};
  bool loaded = false;
};
std::vector<LocalReference> reference_forces;
std::unordered_map<std::int64_t, std::size_t> local_reference_index;
ReferenceFrame reference_frame;
bool reference_ready = false;
double reference_norm2 = 0, accumulated_mse = 0;
std::uint64_t force_samples = 0;
double local_work = 0, previous_work = 0;
rbmd::Id previous_step = 0;
std::string RankSuffix() { return ".rank" + std::to_string(rbmd::mpi::CurrentRank()); }
std::size_t LocalCount() {
  return static_cast<std::size_t>(*DataManager::getInstance().getMDData()->_structure_info_data->_num_atoms);
}
template<class V> auto Owned(const V& v, std::size_t n) {
  if (v.size() < n) throw std::runtime_error("diagnostics: device array shorter than owned atom count");
  return thrust::host_vector<typename V::value_type>(v.begin(), v.begin() + n);
}
double GlobalSum(double local) {
#ifdef USE_MPI
  double global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global;
#else
  return local;
#endif
}
void Open(std::ofstream& out, const std::string& path) {
  out.open(path, std::ios::trunc);
  if (!out) throw std::runtime_error("Cannot create diagnostics file: " + path);
  out << std::setprecision(17);
}
}

void Initialize() {
#ifdef USE_MPI
  // A rank-local opt-in would otherwise change the collective schedule or leave
  // ranks entering observable reductions alone. Check before parsing options.
  const auto cfg = DataManager::getInstance().getConfigData();
  const auto& execution = cfg->GetJsonNode("execution");
  const char* warmup = std::getenv("RBMD_TIMING_WARMUP_STEPS");
  const std::string signature = execution.dump() + (warmup ? warmup : "");
  int length = rbmd::mpi::CurrentRank() == 0 ? static_cast<int>(signature.size()) : 0;
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::string expected = rbmd::mpi::CurrentRank() == 0 ? signature : std::string(length, '\0');
  MPI_Bcast(expected.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
  int mismatch = signature != expected;
  MPI_Allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (mismatch) throw std::runtime_error("execution configuration / timing warmup differs across MPI ranks");
#endif
  const auto& o = GetOptions();
  if (!o.enabled) return;
  const auto config = DataManager::getInstance().getConfigData();
  const auto& exec = config->GetJsonNode("execution");
  const auto& hp = config->GetJsonNode("hyper_parameters");
  const std::string ff = hp.at("force_field").at("type").get<std::string>();
  const std::string neighbor = hp.at("neighbor").at("type").get<std::string>();
  const std::string ensemble = exec.at("ensemble").get<std::string>();
  const std::string unit = config->Get<std::string>("unit", "init_configuration", "read_data");
  if (ff != "CVFF" && ff != "LJ/CUT" && ff != "LJ/CUT/COUL/LONG")
    throw std::runtime_error("revision diagnostics currently support CVFF, LJ/CUT and LJ/CUT/COUL/LONG only");
  if (neighbor != "RBL" && neighbor != "VERLET" && neighbor != "Verlet-List")
    throw std::runtime_error("revision diagnostics require neighbor type RBL or Verlet-List");
  const double dt = exec.at("timestep").get<double>();
  if (!std::isfinite(dt) || dt <= 0) throw std::runtime_error("diagnostics require a finite positive timestep");
  if (ParseUnit(unit) == UNIT::UNKNOWN)
    throw std::runtime_error("diagnostics: unknown energy unit");
  if (hp.contains("coulomb")) {
    const std::string type = hp.at("coulomb").at("type").get<std::string>();
    if (type != "RBE" && type != "EWALD")
      throw std::runtime_error("revision diagnostics support RBE or EWALD Coulomb only");
  }
  if (o.energy && neighbor == "RBL" && hp.at("neighbor").value("energy_rbl_flag", std::string("no")) != "yes")
    throw std::runtime_error("diagnostics.energy requires energy_rbl_flag=yes for RBL; disabled energy is not zero energy");
  if (o.energy && ensemble == "NPT" && !o.fixed_samples)
    throw std::runtime_error("NPT energy diagnostics require separate barostat-work accounting; use NVT/NVE");
  if (o.thermostat_heat) {
    const std::string ctrl = exec.value("temp_ctrl_type", std::string());
    if (ensemble != "NVT" || exec.value("integration_type", std::string()) != "vv" ||
        (ctrl != "NOSE_HOOVER" && ctrl != "BERENDSEN" && ctrl != "RESCALE"))
      throw std::runtime_error("thermostat_heat supports NVT/vv with NOSE_HOOVER, BERENDSEN or RESCALE; Langevin/BM/NPT work is not implemented");
  }
  if (o.unmodified_nve && ensemble != "NVE")
    throw std::runtime_error("unmodified_nve is only valid with ensemble=NVE");
  if (o.energy && ensemble == "NVE" && neighbor == "RBL" && !o.fixed_samples && !o.unmodified_nve)
    throw std::runtime_error("RBL NVE energy diagnostics require unmodified_nve=true to disable the existing energy-stabilization velocity correction");
  if (ensemble == "NVE" && !o.fixed_samples && hp.contains("extend") && hp.at("extend").value("fix_shake", false))
    throw std::runtime_error("Current NVE Solve does not execute SHAKE; constrained-water NVE diagnostics are unsupported");
  reference_forces.clear();
  local_reference_index.clear();
  reference_ready = false;
  reference_norm2 = accumulated_mse = 0;
  force_samples = 0;
  if (!o.reference_force_file.empty() && rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    Open(force_errors, o.prefix + ".force_errors.csv");
    force_errors << "sample,n_atoms,rms_force_error,relative_rms_force_error,cumulative_rms_force_error,"
                    "rms_bias,variance_population,variance_unbiased\n";
  }
  if (o.forces) {
    Open(forces, o.prefix + ".forces" + RankSuffix() + ".csv");
    forces << "sample,id,x,y,z,fx,fy,fz\n";
  }
  if ((o.energy || o.thermostat_heat) && rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    Open(observables, o.prefix + ".observables.csv");
    observables << "step,time,n_atoms,kinetic_total,potential_total,total_energy,thermostat_work_total,thermostat_power_interval\n";
  }
  local_work = previous_work = 0;
  previous_step = 0;
  // Per-rank metadata allows configuration/layout/precision audits without gathers.
  std::ofstream meta;
  Open(meta, o.prefix + ".metadata" + RankSuffix() + ".json");
  nlohmann::ordered_json j;
  j["schema"] = 1;
  j["rank"] = rbmd::mpi::CurrentRank();
  int ranks = 1;
#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#endif
  j["ranks"] = ranks;
  j["real_bytes"] = sizeof(rbmd::Real);
  j["id_bytes"] = sizeof(rbmd::Id);
  j["execution"] = exec;
  j["init_configuration"] = config->GetJsonNode("init_configuration");
  j["outputs"] = config->GetJsonNode("outputs");
  for (const char* key : {"RBMD_FORCE_OVERLAP", "RBMD_BONDED_OVERLAP", "RBMD_TIMING_WARMUP_STEPS",
                          "RBMD_RBL_PAIR_TPP", "RBMD_VL_PAIR_TPP", "RBMD_DETAILED_PAIR_TIMING"}) {
    if (const char* value = std::getenv(key)) j["environment"][key] = value;
  }
  j["hyper_parameters"] = hp;
  j["unit"] = unit;
  j["energy_normalization"] = "system total in configured energy units";
  j["force_id_convention"] = "native RBMD IDs (no added offset), owned atoms only";
  j["force_scope"] = "total force immediately after force evaluation, before thermostat and SHAKE-B";
  if (!o.reference_force_file.empty()) {
    j["reference_force_file"] = o.reference_force_file;
    j["reference_timestep_requested"] = o.reference_timestep;
    j["reference_storage"] = "streamed sorted dump; owned atoms only; bounded MPI ownership checks";
    j["force_statistics"] = "double-precision Welford moments of each atom's force error";
    j["reference_rms_definition"] = "sqrt(sum_atoms |F-Fref|^2 / N); cumulative RMS averages squared errors over samples";
  }
  j["time_axis"] = o.fixed_samples ? "sample index; physical time fixed at zero" : "integration step";
  j["heat_definition"] = "sum of kinetic-energy changes across thermostat velocity-scaling substeps; positive into atoms; power averaged over output interval";
  j["rbe_energy"] = "sampled instantaneous estimator, not a conserved Hamiltonian";
  j["timing_mode"] = o.isolated_timing ? "serialized diagnostic GPU events and inclusive workflow wall time" : "production schedule; overlap spans are not additive";
  j["trajectory"] = "existing trajectory_out output; fixed_samples does not run trajectory output";
  meta << j.dump(2) << '\n';
}

double LocalKineticEnergy() {
  rbmd::revision::DeviceSynchronize();
  const auto d = DataManager::getInstance().getDeviceData();
  const auto n = LocalCount();
  const auto type = Owned(d->_d_atoms_type, n);
  const auto vx = Owned(d->_d_vx, n), vy = Owned(d->_d_vy, n), vz = Owned(d->_d_vz, n);
  const thrust::host_vector<rbmd::Real> mass(d->_d_mass);
  const auto unit = ParseUnit(DataManager::getInstance().getConfigData()->Get<std::string>("unit", "init_configuration", "read_data"));
  const double factor = unit == UNIT::LJ ? UnitFactor<UNIT::LJ>::_mvv2e :
      unit == UNIT::REAL ? UnitFactor<UNIT::REAL>::_mvv2e : UnitFactor<UNIT::METAL>::_mvv2e;
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (type[i] < 0 || static_cast<std::size_t>(type[i]) >= mass.size())
      throw std::runtime_error("diagnostics: invalid mass type index");
    const double x = vx[i], y = vy[i], z = vz[i];
    sum += .5 * factor * double(mass[type[i]]) * (x*x + y*y + z*z);
  }
  return sum;
}
void AddThermostatWork(double work) { local_work += work; }

void CaptureForces(rbmd::Id step) {
  const auto& o = GetOptions();
  if ((!o.forces && o.reference_force_file.empty()) || step % o.interval || (o.fixed_samples && step == 0)) return;
  rbmd::revision::DeviceSynchronize();
  const auto d = DataManager::getInstance().getDeviceData();
  const auto n = LocalCount();
  const auto ids = Owned(d->_d_atoms_id, n);
  const auto x = Owned(d->_d_px, n), y = Owned(d->_d_py, n), z = Owned(d->_d_pz, n);
  const auto fx = Owned(d->_d_fx, n), fy = Owned(d->_d_fy, n), fz = Owned(d->_d_fz, n);
  if (o.forces) {
    for (std::size_t i = 0; i < n; ++i)
      forces << step << ',' << ids[i] << ',' << x[i] << ',' << y[i] << ',' << z[i]
             << ',' << fx[i] << ',' << fy[i] << ',' << fz[i] << '\n';
    forces.flush();
    if (!forces) throw std::runtime_error("Failed writing force diagnostics");
  }
  if (!o.reference_force_file.empty()) {
    // Load after force initialization: owned atom IDs and wrapped coordinates now exist.
    if (!reference_ready) {
      reference_forces.resize(n);
      local_reference_index.reserve(n);
      bool unique = true;
      for (std::size_t i = 0; i < n; ++i) {
        unique &= local_reference_index.emplace(ids[i], i).second;
        reference_forces[i].frozen_position = {{double(x[i]), double(y[i]), double(z[i])}};
      }
      if (GlobalSum(unique ? 0 : 1)) throw std::runtime_error("Duplicate local owned atom ID");
      std::ifstream input(o.reference_force_file);
      if (GlobalSum(input ? 0 : 1)) throw std::runtime_error("Cannot open reference force dump on all ranks");
      // Every rank streams the shared file once. Only a bounded ownership block is reduced,
      // never a full dump or an N-entry ownership array on every sample.
      std::vector<int> owners;
      owners.reserve(16384);
      auto check_owners = [&]() {
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, owners.data(), static_cast<int>(owners.size()),
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        for (int count : owners)
          if (count != 1) throw std::runtime_error("Reference/sample ID sets differ or MPI ownership is duplicated");
        owners.clear();
      };
      reference_frame = VisitReferenceDump(input, o.reference_timestep, o.reference_id_offset,
          [&](std::int64_t id, const ReferenceForce& ref) {
            const auto it = local_reference_index.find(id);
            owners.push_back(it != local_reference_index.end());
            if (it != local_reference_index.end()) {
              auto& entry = reference_forces[it->second];
              entry.reference = ref;
              entry.loaded = true;
              for (double f : ref.force) reference_norm2 += f*f;
            }
            if (owners.size() == 16384) check_owners();
          });
      if (!owners.empty()) check_owners();
      bool complete = true;
      for (const auto& entry : reference_forces) complete &= entry.loaded;
      if (GlobalSum(complete ? 0 : 1) || GlobalSum(double(n)) != reference_frame.count)
        throw std::runtime_error("Reference/sample atom sets differ");
      reference_norm2 = GlobalSum(reference_norm2) / reference_frame.count;
      if (!std::isfinite(reference_norm2)) throw std::runtime_error("Reference force norm overflow");
      reference_ready = true;
    }
    double error2 = 0, bias2 = 0, m2 = 0;
    bool valid = n == reference_forces.size();
    const auto sample = force_samples + 1;
    for (std::size_t i = 0; i < n; ++i) {
      const auto it = local_reference_index.find(ids[i]);
      if (it == local_reference_index.end()) { valid = false; continue; }
      auto& entry = reference_forces[it->second];
      // Frozen sampling permits reordering, but cannot change ownership, positions or ID sets.
      if (entry.moments.samples != force_samples) { valid = false; continue; }
      const std::array<double, 3> position{{double(x[i]), double(y[i]), double(z[i])}};
      const std::array<double, 3> force{{double(fx[i]), double(fy[i]), double(fz[i])}};
      std::array<double, 3> error;
      bool atom_valid = true;
      for (int d = 0; d < 3; ++d) {
        atom_valid &= std::isfinite(force[d]) && position[d] == entry.frozen_position[d] &&
          ReferencePositionMatches(reference_frame, d, position[d], entry.reference.position[d],
                                   o.reference_position_tolerance);
        error[d] = force[d] - entry.reference.force[d];
      }
      if (!atom_valid) { valid = false; continue; }
      entry.moments.Add(error);
      for (double delta : error) error2 += delta*delta;
      bias2 += entry.moments.BiasSquared();
      m2 += entry.moments.m2;
    }
    for (const auto& entry : reference_forces) valid &= entry.moments.samples == sample;
    if (GlobalSum(valid ? 0 : 1))
      throw std::runtime_error("Fixed force sample has invalid IDs, moved coordinates, reference mismatch or nonfinite force");
    const double count = double(reference_frame.count);
    const double mse = GlobalSum(error2) / count;
    bias2 = GlobalSum(bias2) / count;
    m2 = GlobalSum(m2) / count;
    accumulated_mse += mse;
    if (!std::isfinite(accumulated_mse) || !std::isfinite(bias2) || !std::isfinite(m2))
      throw std::runtime_error("Accumulated force statistics overflow");
    force_samples = sample;
    if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
      force_errors << step << ',' << reference_frame.count << ',' << std::sqrt(mse) << ','
                   << (reference_norm2 > 0 ? std::sqrt(mse/reference_norm2) : std::numeric_limits<double>::quiet_NaN())
                   << ',' << std::sqrt(accumulated_mse/force_samples)
                   << ',' << std::sqrt(bias2) << ',' << m2/force_samples << ',';
      if (force_samples > 1) force_errors << m2/(force_samples-1);
      force_errors << '\n';
      force_errors.flush();
      if (!force_errors) throw std::runtime_error("Failed writing online force errors");
      if (step == o.fixed_samples) {
        std::ofstream out;
        Open(out, o.prefix + ".force_statistics.json");
        nlohmann::ordered_json j;
        j["schema"] = 1;
        j["n_atoms"] = reference_frame.count;
        j["n_samples"] = force_samples;
        j["mse"] = accumulated_mse/force_samples;
        j["rms_force_error"] = std::sqrt(accumulated_mse/force_samples);
        if (reference_norm2 > 0) j["relative_rms_force_error"] = std::sqrt(accumulated_mse/force_samples/reference_norm2);
        else j["relative_rms_force_error"] = nullptr;
        j["rms_bias"] = std::sqrt(bias2);
        j["variance_population"] = m2/force_samples;
        if (force_samples > 1) j["variance_unbiased"] = m2/(force_samples-1);
        else j["variance_unbiased"] = nullptr;
        j["reference_force_file"] = o.reference_force_file;
        j["reference_timestep"] = reference_frame.step;
        j["reference_mean_squared_force"] = reference_norm2;
        j["reference_position_tolerance"] = o.reference_position_tolerance;
        j["reference_id_offset"] = o.reference_id_offset;
        j["position_check"] = "periodic orthogonal reference; exact frozen sampled positions";
        j["reference_storage"] = "owned atoms only; streamed sorted text";
        out << j.dump(2) << '\n';
        if (!out) throw std::runtime_error("Failed writing online force statistics");
      }
    }
  }
}

void CaptureObservables(rbmd::Id step) {
  const auto& o = GetOptions();
  if ((!o.energy && !o.thermostat_heat) || step % o.interval || (o.fixed_samples && step == 0)) return;
  const double n = GlobalSum(static_cast<double>(LocalCount()));
  const double kinetic = GlobalSum(LocalKineticEnergy());
  const double work = GlobalSum(local_work);
  const auto config = DataManager::getInstance().getConfigData();
  const double dt = config->Get<double>("timestep", "execution");
  const double elapsed = double(step - previous_step) * dt;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  double potential = nan;
  if (o.energy) {
    potential = ThermoStats::Instance().GetThermoData("total-potential-energy");
    const std::string ff = config->Get<std::string>("type", "hyper_parameters", "force_field");
    const auto unit = ParseUnit(config->Get<std::string>("unit", "init_configuration", "read_data"));
    // Legacy reporting uses per-atom LJ energy, and CVFF normalizes only kspace in LJ units.
    if (ff == "LJ/CUT" || (ff == "LJ/CUT/COUL/LONG" && unit == UNIT::LJ)) potential *= n;
    else if (ff == "CVFF" && unit == UNIT::LJ && config->PathExists({"hyper_parameters", "coulomb"}))
      potential += (n - 1) * ThermoStats::Instance().GetThermoData("kspace");
  }
  if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
    observables << step << ',' << (o.fixed_samples ? 0 : double(step)*dt) << ',' << n << ','
        << kinetic << ',' << potential << ',' << kinetic + potential << ','
        << (o.thermostat_heat ? work : nan) << ','
        << (o.thermostat_heat && elapsed > 0 ? (work-previous_work)/elapsed : nan) << '\n';
    observables.flush();
    if (!observables) throw std::runtime_error("Failed writing observable diagnostics");
  }
  previous_step = step;
  previous_work = work;
}
void WriteTiming() {
  const auto& o = GetOptions();
  if (o.enabled) TimingStatistics::Instance().write_csv(o.prefix + ".timing" + RankSuffix() + ".csv");
}
}}
