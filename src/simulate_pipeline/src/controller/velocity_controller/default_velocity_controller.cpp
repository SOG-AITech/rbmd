#include "default_velocity_controller.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include "common/mpi_reduce_helper.hpp"
#include "data_manager.h"
#include "device_types.h"
#include "neighbor_list/include/linked_cell/linked_cell_locator.h"
#include "output/include/Logger.hpp"
#include "simulate.h"
#include "unit_factor.h"
#include "update_velocity_op.h"

DefaultVelocityController::DefaultVelocityController(){};

namespace {
bool VelocityDebugEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("RBMD_DEBUG_VELOCITY_STATS");
    return env != nullptr && std::string(env) != "0" &&
           std::string(env) != "false" && std::string(env) != "FALSE";
  }();
  return enabled;
}

int VelocityDebugEvery() {
  static const int every = [] {
    const char* env = std::getenv("RBMD_DEBUG_VELOCITY_STATS_EVERY");
    if (env == nullptr) {
      return 1;
    }
    return std::max(1, std::atoi(env));
  }();
  return every;
}

rbmd::Real MaxAbsComponent(const thrust::device_vector<rbmd::Real>& values) {
  if (values.empty()) {
    return 0.0;
  }
  auto max_it = thrust::max_element(values.begin(), values.end());
  auto min_it = thrust::min_element(values.begin(), values.end());
  const rbmd::Real max_val = *max_it;
  const rbmd::Real min_val = *min_it;
  return std::max(std::abs(max_val), std::abs(min_val));
}

void EmitVelocityDebugStats(const DeviceData& device_data, rbmd::Id local_num_atoms,
                            const char* stage) {
  if (!VelocityDebugEnabled() || local_num_atoms <= 0) {
    return;
  }
  if (test_current_step % VelocityDebugEvery() != 0) {
    return;
  }

  const rbmd::Real local_max_abs_vx = MaxAbsComponent(device_data._d_vx);
  const rbmd::Real local_max_abs_vy = MaxAbsComponent(device_data._d_vy);
  const rbmd::Real local_max_abs_vz = MaxAbsComponent(device_data._d_vz);
  const rbmd::Real local_max_abs_fx = MaxAbsComponent(device_data._d_fx);
  const rbmd::Real local_max_abs_fy = MaxAbsComponent(device_data._d_fy);
  const rbmd::Real local_max_abs_fz = MaxAbsComponent(device_data._d_fz);

  rbmd::Real debug_stats[6] = {
      local_max_abs_vx, local_max_abs_vy, local_max_abs_vz,
      local_max_abs_fx, local_max_abs_fy, local_max_abs_fz};
  AllReduceRealBufferInPlace(debug_stats, 6);

  thrust::host_vector<rbmd::Id> h_atoms_type(local_num_atoms);
  thrust::copy(device_data._d_atoms_type.begin(),
               device_data._d_atoms_type.begin() + local_num_atoms,
               h_atoms_type.begin());
  thrust::host_vector<rbmd::Id> h_atoms_id(local_num_atoms);
  thrust::copy(device_data._d_atoms_id.begin(),
               device_data._d_atoms_id.begin() + local_num_atoms,
               h_atoms_id.begin());
  thrust::host_vector<rbmd::Real> h_mass = device_data._d_mass;

  rbmd::Id local_min_type = h_atoms_type.empty() ? 0 : h_atoms_type[0];
  rbmd::Id local_max_type = h_atoms_type.empty() ? 0 : h_atoms_type[0];
  rbmd::Id local_invalid_type_count = 0;
  rbmd::Id local_nonpositive_mass_count = 0;
  rbmd::Id local_first_bad_atom_idx = -1;
  rbmd::Id local_first_bad_atom_id = -1;
  rbmd::Id local_first_bad_type = -1;
  rbmd::Real local_first_bad_mass = 0.0;
  rbmd::Id local_nonfinite_velocity_count = 0;
  rbmd::Id local_nonfinite_force_count = 0;
  rbmd::Id local_large_velocity_count = 0;
  rbmd::Id local_first_large_velocity_atom_idx = -1;
  rbmd::Id local_first_large_velocity_atom_id = -1;
  rbmd::Id local_first_large_velocity_type = -1;
  rbmd::Real local_first_large_velocity_mass = 0.0;
  rbmd::Real local_first_large_vx = 0.0;
  rbmd::Real local_first_large_vy = 0.0;
  rbmd::Real local_first_large_vz = 0.0;
  rbmd::Real local_first_large_fx = 0.0;
  rbmd::Real local_first_large_fy = 0.0;
  rbmd::Real local_first_large_fz = 0.0;
  const rbmd::Id mass_size = static_cast<rbmd::Id>(h_mass.size());
  thrust::host_vector<rbmd::Real> h_vx(local_num_atoms);
  thrust::host_vector<rbmd::Real> h_vy(local_num_atoms);
  thrust::host_vector<rbmd::Real> h_vz(local_num_atoms);
  thrust::host_vector<rbmd::Real> h_fx(local_num_atoms);
  thrust::host_vector<rbmd::Real> h_fy(local_num_atoms);
  thrust::host_vector<rbmd::Real> h_fz(local_num_atoms);
  thrust::copy(device_data._d_vx.begin(),
               device_data._d_vx.begin() + local_num_atoms, h_vx.begin());
  thrust::copy(device_data._d_vy.begin(),
               device_data._d_vy.begin() + local_num_atoms, h_vy.begin());
  thrust::copy(device_data._d_vz.begin(),
               device_data._d_vz.begin() + local_num_atoms, h_vz.begin());
  thrust::copy(device_data._d_fx.begin(),
               device_data._d_fx.begin() + local_num_atoms, h_fx.begin());
  thrust::copy(device_data._d_fy.begin(),
               device_data._d_fy.begin() + local_num_atoms, h_fy.begin());
  thrust::copy(device_data._d_fz.begin(),
               device_data._d_fz.begin() + local_num_atoms, h_fz.begin());
  for (rbmd::Id i = 0; i < local_num_atoms; ++i) {
    const rbmd::Id type = h_atoms_type[static_cast<std::size_t>(i)];
    local_min_type = std::min(local_min_type, type);
    local_max_type = std::max(local_max_type, type);
    const bool invalid_type = type < 0 || type >= mass_size;
    rbmd::Real mass_value = 0.0;
    if (!invalid_type) {
      mass_value = h_mass[static_cast<std::size_t>(type)];
    }
    const bool bad_mass = invalid_type || !std::isfinite(mass_value) ||
                          mass_value <= 0.0;
    if (invalid_type) {
      ++local_invalid_type_count;
    }
    if (bad_mass) {
      ++local_nonpositive_mass_count;
      if (local_first_bad_atom_idx < 0) {
        local_first_bad_atom_idx = i;
        local_first_bad_atom_id = h_atoms_id[static_cast<std::size_t>(i)];
        local_first_bad_type = type;
        local_first_bad_mass = mass_value;
      }
    }

    const rbmd::Real vx = h_vx[static_cast<std::size_t>(i)];
    const rbmd::Real vy = h_vy[static_cast<std::size_t>(i)];
    const rbmd::Real vz = h_vz[static_cast<std::size_t>(i)];
    const rbmd::Real fx = h_fx[static_cast<std::size_t>(i)];
    const rbmd::Real fy = h_fy[static_cast<std::size_t>(i)];
    const rbmd::Real fz = h_fz[static_cast<std::size_t>(i)];
    const bool nonfinite_velocity = !std::isfinite(vx) || !std::isfinite(vy) ||
                                    !std::isfinite(vz);
    const bool nonfinite_force = !std::isfinite(fx) || !std::isfinite(fy) ||
                                 !std::isfinite(fz);
    if (nonfinite_velocity) {
      ++local_nonfinite_velocity_count;
    }
    if (nonfinite_force) {
      ++local_nonfinite_force_count;
    }
    const bool large_velocity =
        std::abs(vx) > 1.0e3 || std::abs(vy) > 1.0e3 || std::abs(vz) > 1.0e3;
    if (large_velocity) {
      ++local_large_velocity_count;
      if (local_first_large_velocity_atom_idx < 0) {
        local_first_large_velocity_atom_idx = i;
        local_first_large_velocity_atom_id = h_atoms_id[static_cast<std::size_t>(i)];
        local_first_large_velocity_type = type;
        local_first_large_velocity_mass = mass_value;
        local_first_large_vx = vx;
        local_first_large_vy = vy;
        local_first_large_vz = vz;
        local_first_large_fx = fx;
        local_first_large_fy = fy;
        local_first_large_fz = fz;
      }
    }
  }

  Logger::Instance().info(
      "[velocity_debug] step={} stage={} max|v|=({}, {}, {}) max|f|=({}, {}, {}) "
      "local_num_atoms={} mass_size={} atom_type_range=[{}, {}] invalid_type_count={} "
      "bad_mass_count={} first_bad_atom_idx={} first_bad_atom_id={} first_bad_type={} "
      "first_bad_mass={} nonfinite_velocity_count={} nonfinite_force_count={} "
      "large_velocity_count={} first_large_velocity_atom_idx={} "
      "first_large_velocity_atom_id={} first_large_velocity_type={} "
      "first_large_velocity_mass={} first_large_v=({}, {}, {}) first_large_f=({}, {}, {})",
      test_current_step, stage, debug_stats[0], debug_stats[1], debug_stats[2],
      debug_stats[3], debug_stats[4], debug_stats[5], local_num_atoms, mass_size,
      local_min_type, local_max_type, local_invalid_type_count,
      local_nonpositive_mass_count, local_first_bad_atom_idx,
      local_first_bad_atom_id, local_first_bad_type, local_first_bad_mass,
      local_nonfinite_velocity_count, local_nonfinite_force_count,
      local_large_velocity_count, local_first_large_velocity_atom_idx,
      local_first_large_velocity_atom_id, local_first_large_velocity_type,
      local_first_large_velocity_mass, local_first_large_vx, local_first_large_vy,
      local_first_large_vz, local_first_large_fx, local_first_large_fy,
      local_first_large_fz);
}
}  // namespace

void DefaultVelocityController::Init() {

  auto& num_atoms = *(_structure_info_data->_num_atoms);

  _dt = DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
          "timestep", "execution");//0.001
  auto unit  = DataManager::getInstance().getConfigData()->Get
    <std::string>("unit", "init_configuration", "read_data");
  UNIT unit_factor = ParseUnit(unit);

  switch (unit_factor) {
    case UNIT::METAL:
      _fmt2v = UnitFactor<UNIT::METAL>::_fmt2v;
      break;
    case UNIT::LJ:
      _fmt2v = UnitFactor<UNIT::LJ>::_fmt2v;
      break;
    case UNIT::REAL:
      _fmt2v = UnitFactor<UNIT::REAL>::_fmt2v;
      break;
    default:
      break;
  }

  //
  const auto& config = DataManager::getInstance().getConfigData();
  auto integration_type = config->Get<std::string>("integration_type", "execution");
  if("bm" ==integration_type) {
    _par_a = DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
        "par_a", "execution");
    _par_b = DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
            "par_b", "execution");

    _d_prev_fx.resize(num_atoms, 0.0);
    _d_prev_fy.resize(num_atoms, 0.0);
    _d_prev_fz.resize(num_atoms, 0.0);
    _d_pr_prev_fx.resize(num_atoms, 0.0);
    _d_pr_prev_fy.resize(num_atoms, 0.0);
    _d_pr_prev_fz.resize(num_atoms, 0.0);
  }
}

void DefaultVelocityController::Update() {
  const rbmd::Id num_atoms = *(_structure_info_data->_num_atoms);
  EmitVelocityDebugStats(*_device_data, num_atoms,
                         "before_half_kick");
  bool shake = DataManager::getInstance().getConfigData()->GetJudge<bool>
  ( "fix_shake", "hyper_parameters", "extend");
  if (shake) {
      thrust::copy(_device_data->_d_vx.begin(),_device_data->_d_vx.begin() + num_atoms,
        _device_data->_d_shake_vx.begin());
      thrust::copy(_device_data->_d_vy.begin(),_device_data->_d_vy.begin() + num_atoms,
        _device_data->_d_shake_vy.begin());
      thrust::copy(_device_data->_d_vz.begin(),_device_data->_d_vz.begin() + num_atoms,
        _device_data->_d_shake_vz.begin());
  }

  op::UpdateVelocityOp<device::DEVICE_GPU>()(
      num_atoms, _dt, _fmt2v,
      thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_mass.data()),
      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
      thrust::raw_pointer_cast(_device_data->_d_fz.data()),
      thrust::raw_pointer_cast(_device_data->_d_vx.data()),
      thrust::raw_pointer_cast(_device_data->_d_vy.data()),
      thrust::raw_pointer_cast(_device_data->_d_vz.data()));
  EmitVelocityDebugStats(*_device_data, num_atoms,
                         "after_half_kick");

}

void DefaultVelocityController::Updatebm(){
   //std::cout << "test_current_step--v: "  << test_current_step <<  std::endl;
  op::UpdateVelocityOpbm<device::DEVICE_GPU>()(
      *(_structure_info_data->_num_atoms), _par_a,_par_b, _dt, test_current_step, _fmt2v,
      thrust::raw_pointer_cast(_device_data->_d_atoms_type.data()),
      thrust::raw_pointer_cast(_device_data->_d_mass.data()),
      thrust::raw_pointer_cast(_device_data->_d_fx.data()),
      thrust::raw_pointer_cast(_device_data->_d_fy.data()),
      thrust::raw_pointer_cast(_device_data->_d_fz.data()),
      thrust::raw_pointer_cast(_d_prev_fx.data()),
      thrust::raw_pointer_cast(_d_prev_fy.data()),
      thrust::raw_pointer_cast(_d_prev_fz.data()),
      thrust::raw_pointer_cast(_d_pr_prev_fx.data()),
      thrust::raw_pointer_cast(_d_pr_prev_fy.data()),
      thrust::raw_pointer_cast(_d_pr_prev_fz.data()),
      thrust::raw_pointer_cast(_device_data->_d_vx.data()),
      thrust::raw_pointer_cast(_device_data->_d_vy.data()),
      thrust::raw_pointer_cast(_device_data->_d_vz.data()));
}
