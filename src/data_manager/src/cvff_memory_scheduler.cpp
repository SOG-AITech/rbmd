#include "./include/scheduler/cvff_memory_scheduler.h"

#include <cmath>
#include <sstream>
#if defined(USE_MPI) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <mpi.h>
#define RBMD_HOST_MPI_AVAILABLE 1
#else
#define RBMD_HOST_MPI_AVAILABLE 0
#endif

#include "data_manager.h"
#include <cstddef>
#include <stdexcept>

namespace {

int CurrentRank() {
#if RBMD_HOST_MPI_AVAILABLE
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      return rank;
    }
  }
#endif
  return 0;
}

std::string FormatBondCoeffSample(const rbmd::Real* values, std::size_t count,
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

void ValidateBondCoeffArrayOrThrow(const char* stage, const rbmd::Real* values,
                                   std::size_t count) {
  if (stage == nullptr || values == nullptr || count == 0) {
    return;
  }

  double max_value = 0.0;
  std::size_t max_index = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = values[i];
    if (!std::isfinite(value)) {
      std::ostringstream oss;
      oss << "[rank " << CurrentRank()
          << "] CVFFMemoryScheduler observed non-finite bond_equilibrium during "
          << stage
          << ", ptr=" << static_cast<const void*>(values)
          << ", count=" << count
          << ", sample=" << FormatBondCoeffSample(values, count);
      throw std::runtime_error(oss.str());
    }
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  if (max_value > 10.0) {
    std::ostringstream oss;
    oss << "[rank " << CurrentRank()
        << "] CVFFMemoryScheduler observed suspicious bond_equilibrium during "
        << stage
        << ": max=" << max_value << " at index " << max_index
        << ", ptr=" << static_cast<const void*>(values)
        << ", count=" << count
        << ", sample=" << FormatBondCoeffSample(values, count);
    throw std::runtime_error(oss.str());
  }
}

}  // namespace
#undef RBMD_HOST_MPI_AVAILABLE

bool CVFFMemoryScheduler::asyncMemoryH2D() {
  if (false == MemoryScheduler::asyncMemoryH2D()) {
    // log
    return false;
  }

  auto& num_atoms_type = *(_structure_info_data->_num_atoms_type);
  auto& num_atoms = *(_structure_info_data->_num_atoms);
  auto& num_bonds = *(_structure_info_data->_num_bonds);
  auto& num_bonds_type = *(_structure_info_data->_num_bonds_type);
  auto& num_angles = *(_structure_info_data->_num_angles);
  auto& num_angles_type = *(_structure_info_data->_num_angles_type);
  auto& num_dihedrals = *(_structure_info_data->_num_dihedrals);
  auto& num_dihedrals_type = *(_structure_info_data->_num_dihedrals_type);
  auto& num_impropers = *(_structure_info_data->_num_impropers);
  auto& num_impropers_type = *(_structure_info_data->_num_impropers_type);
  auto sd = std::dynamic_pointer_cast<FullStructureData>(_structure_data);
  auto fd = std::dynamic_pointer_cast<CVFFForceFieldData>(_force_field_data);

  /// copy data
  _device_data->_d_molecular_id.resize(num_atoms);

  _device_data->_d_bond_type.resize(num_bonds);
  _device_data->_d_bond_id0.resize(num_bonds);
  _device_data->_d_bond_id1.resize(num_bonds);

  _device_data->_d_angle_type.resize(num_angles);
  _device_data->_d_angle_id0.resize(num_angles);
  _device_data->_d_angle_id1.resize(num_angles);
  _device_data->_d_angle_id2.resize(num_angles);
  _device_data->_d_angle_id_vec.resize(num_angles);

  _device_data->_d_dihedral_type.resize(num_dihedrals);
  _device_data->_d_dihedral_id0.resize(num_dihedrals);
  _device_data->_d_dihedral_id1.resize(num_dihedrals);
  _device_data->_d_dihedral_id2.resize(num_dihedrals);
  _device_data->_d_dihedral_id3.resize(num_dihedrals);

  _device_data->_d_improper_type.resize(num_impropers);
  _device_data->_d_improper_id0.resize(num_impropers);
  _device_data->_d_improper_id1.resize(num_impropers);
  _device_data->_d_improper_id2.resize(num_impropers);
  _device_data->_d_improper_id3.resize(num_impropers);

  _device_data->_d_charge.resize(num_atoms);

  _device_data->_d_atoms_vec.resize(sd->_num_atoms_vec_gro);
  _device_data->_d_atoms_count.resize(sd->_num_count_vector);
  _device_data->_d_atoms_offset.resize(sd->_num_atoms_offset );

  _device_data->_d_special_weights.resize(sd->_num_special_weights);
  _device_data->_d_special_ids.resize(sd->_num_special_ids);
  _device_data->_d_special_offsets.resize(sd->_num_special_offsets);
  _device_data->_d_special_count.resize(sd->_num_special_offset_count);

  /// charge
  thrust::copy(sd->_h_charge, sd->_h_charge + num_atoms,
      _device_data->_d_charge.begin());

  /// molecular id
  thrust::copy(sd->_h_molecules_id, sd->_h_molecules_id + num_atoms,
               _device_data->_d_molecular_id.begin());
  /// bond
  thrust::copy(sd->_h_bond_type, sd->_h_bond_type + num_bonds,
               _device_data->_d_bond_type.begin());
  thrust::copy(sd->_h_bond_id0, sd->_h_bond_id0 + num_bonds,
               _device_data->_d_bond_id0.begin());
  thrust::copy(sd->_h_bond_id1, sd->_h_bond_id1 + num_bonds,
               _device_data->_d_bond_id1.begin());

  //special weights and ids
  thrust::copy(sd->_h_special_weights, sd->_h_special_weights + sd->_num_special_weights,
      _device_data->_d_special_weights.begin());
  thrust::copy(sd->_h_special_ids, sd->_h_special_ids + sd->_num_special_ids,
      _device_data->_d_special_ids.begin());
  thrust::copy(sd->_h_special_offsets, sd->_h_special_offsets + sd->_num_special_offsets,
      _device_data->_d_special_offsets.begin());
  thrust::copy(sd->_h_special_offset_count, sd->_h_special_offset_count +
    sd->_num_special_offset_count,_device_data->_d_special_count.begin());

  // special atoms_vec
  thrust::copy(sd->_h_atoms_vec_gro, sd->_h_atoms_vec_gro + sd->_num_atoms_vec_gro,
      _device_data->_d_atoms_vec.begin());

  thrust::copy(sd->_h_count_vector, sd->_h_count_vector + sd->_num_count_vector,
      _device_data->_d_atoms_count.begin());

  thrust::copy(sd->_h_atoms_offset, sd->_h_atoms_offset + sd->_num_atoms_offset,
  _device_data->_d_atoms_offset.begin());

  // per-atom topology (new layout for special bond parallel)
  if (_device_data->nmax > 0) {
    // Call-chain contract for Task7A:
    // reader::mpi::ConvertGlobalToPerAtomTopology fills host per-atom arrays;
    // scheduler uploads them so CVFF per-atom force ops can consume d_num_* and d_*_atom*.
    _device_data->ResizeTopology(_device_data->nmax);
    const std::size_t nmax = static_cast<std::size_t>(_device_data->nmax);

    const int bond_per_atom =
        _device_data->bond_per_atom > 0 ? _device_data->bond_per_atom : 0;
    const int angle_per_atom =
        _device_data->angle_per_atom > 0 ? _device_data->angle_per_atom : 0;
    const int dihedral_per_atom =
        _device_data->dihedral_per_atom > 0 ? _device_data->dihedral_per_atom : 0;
    const int improper_per_atom =
        _device_data->improper_per_atom > 0 ? _device_data->improper_per_atom : 0;

#ifdef USE_MPI
    auto require_positive_per_atom = [](bool has_topology, int per_atom,
                                        const char* name) {
      if (has_topology && per_atom <= 0) {
        throw std::runtime_error(
            std::string("MPI topology upload requires positive ") + name +
            "_per_atom");
      }
    };
    require_positive_per_atom(num_bonds > 0, bond_per_atom, "bond");
    require_positive_per_atom(num_angles > 0, angle_per_atom, "angle");
    require_positive_per_atom(num_dihedrals > 0, dihedral_per_atom, "dihedral");
    require_positive_per_atom(num_impropers > 0, improper_per_atom, "improper");
#endif

    if (sd->_h_num_bond) {
      thrust::copy(sd->_h_num_bond, sd->_h_num_bond + nmax,
                   _device_data->d_num_bond.begin());
    }
    if (sd->_h_bond_type_per_atom && bond_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(bond_per_atom);
      thrust::copy(sd->_h_bond_type_per_atom, sd->_h_bond_type_per_atom + size,
                   _device_data->d_bond_type.begin());
    }
    if (sd->_h_bond_atom_per_atom && bond_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(bond_per_atom);
      thrust::copy(sd->_h_bond_atom_per_atom, sd->_h_bond_atom_per_atom + size,
                   _device_data->d_bond_atom.begin());
    }

    if (sd->_h_num_angle) {
      thrust::copy(sd->_h_num_angle, sd->_h_num_angle + nmax,
                   _device_data->d_num_angle.begin());
    }
    if (sd->_h_angle_type_per_atom && angle_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(angle_per_atom);
      thrust::copy(sd->_h_angle_type_per_atom, sd->_h_angle_type_per_atom + size,
                   _device_data->d_angle_type.begin());
    }
    if (sd->_h_angle_atom1_per_atom && angle_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(angle_per_atom);
      thrust::copy(sd->_h_angle_atom1_per_atom, sd->_h_angle_atom1_per_atom + size,
                   _device_data->d_angle_atom1.begin());
    }
    if (sd->_h_angle_atom2_per_atom && angle_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(angle_per_atom);
      thrust::copy(sd->_h_angle_atom2_per_atom, sd->_h_angle_atom2_per_atom + size,
                   _device_data->d_angle_atom2.begin());
    }
    if (sd->_h_angle_atom3_per_atom && angle_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(angle_per_atom);
      thrust::copy(sd->_h_angle_atom3_per_atom, sd->_h_angle_atom3_per_atom + size,
                   _device_data->d_angle_atom3.begin());
    }

    if (sd->_h_num_dihedral) {
      thrust::copy(sd->_h_num_dihedral, sd->_h_num_dihedral + nmax,
                   _device_data->d_num_dihedral.begin());
    }
    if (sd->_h_dihedral_type_per_atom && dihedral_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(dihedral_per_atom);
      thrust::copy(sd->_h_dihedral_type_per_atom, sd->_h_dihedral_type_per_atom + size,
                   _device_data->d_dihedral_type.begin());
    }
    if (sd->_h_dihedral_atom1_per_atom && dihedral_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(dihedral_per_atom);
      thrust::copy(sd->_h_dihedral_atom1_per_atom, sd->_h_dihedral_atom1_per_atom + size,
                   _device_data->d_dihedral_atom1.begin());
    }
    if (sd->_h_dihedral_atom2_per_atom && dihedral_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(dihedral_per_atom);
      thrust::copy(sd->_h_dihedral_atom2_per_atom, sd->_h_dihedral_atom2_per_atom + size,
                   _device_data->d_dihedral_atom2.begin());
    }
    if (sd->_h_dihedral_atom3_per_atom && dihedral_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(dihedral_per_atom);
      thrust::copy(sd->_h_dihedral_atom3_per_atom, sd->_h_dihedral_atom3_per_atom + size,
                   _device_data->d_dihedral_atom3.begin());
    }
    if (sd->_h_dihedral_atom4_per_atom && dihedral_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(dihedral_per_atom);
      thrust::copy(sd->_h_dihedral_atom4_per_atom, sd->_h_dihedral_atom4_per_atom + size,
                   _device_data->d_dihedral_atom4.begin());
    }

    if (sd->_h_num_improper) {
      thrust::copy(sd->_h_num_improper, sd->_h_num_improper + nmax,
                   _device_data->d_num_improper.begin());
    }
    if (sd->_h_improper_type_per_atom && improper_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(improper_per_atom);
      thrust::copy(sd->_h_improper_type_per_atom, sd->_h_improper_type_per_atom + size,
                   _device_data->d_improper_type.begin());
    }
    if (sd->_h_improper_atom1_per_atom && improper_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(improper_per_atom);
      thrust::copy(sd->_h_improper_atom1_per_atom, sd->_h_improper_atom1_per_atom + size,
                   _device_data->d_improper_atom1.begin());
    }
    if (sd->_h_improper_atom2_per_atom && improper_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(improper_per_atom);
      thrust::copy(sd->_h_improper_atom2_per_atom, sd->_h_improper_atom2_per_atom + size,
                   _device_data->d_improper_atom2.begin());
    }
    if (sd->_h_improper_atom3_per_atom && improper_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(improper_per_atom);
      thrust::copy(sd->_h_improper_atom3_per_atom, sd->_h_improper_atom3_per_atom + size,
                   _device_data->d_improper_atom3.begin());
    }
    if (sd->_h_improper_atom4_per_atom && improper_per_atom > 0) {
      const std::size_t size = nmax * static_cast<std::size_t>(improper_per_atom);
      thrust::copy(sd->_h_improper_atom4_per_atom, sd->_h_improper_atom4_per_atom + size,
                   _device_data->d_improper_atom4.begin());
    }

    // Special bond per-atom topology
    if (sd->_h_nspecial && sd->_h_special && sd->_h_maxspecial > 0) {
      const int maxspecial = sd->_h_maxspecial;
      const rbmd::Id nmax_special =
          sd->_h_nmax_special > 0
              ? static_cast<rbmd::Id>(sd->_h_nmax_special)
              : static_cast<rbmd::Id>(nmax);

      if (_device_data->maxspecial != maxspecial || _device_data->nmax != nmax_special) {
        _device_data->maxspecial = maxspecial;
        _device_data->nmax = nmax_special;
        _device_data->ResizeTopology(nmax_special);
      }

      const std::size_t nspecial_size =
          static_cast<std::size_t>(nmax_special) * 3;
      if (_device_data->d_nspecial.size() >= nspecial_size) {
        thrust::copy(sd->_h_nspecial, sd->_h_nspecial + nspecial_size,
                     _device_data->d_nspecial.begin());
      }

      const std::size_t special_size =
          static_cast<std::size_t>(nmax_special) * static_cast<std::size_t>(maxspecial);
      if (_device_data->d_special.size() >= special_size) {
        thrust::copy(sd->_h_special, sd->_h_special + special_size,
                     _device_data->d_special.begin());
      }
    }
  }

  //GPU
  // thrust::device_vector<rbmd::Id> d_atoms_offset_temp(sd->_h_countVector.size());
  // thrust::copy(sd->_h_countVector.begin(), sd->_h_countVector.end(),
  //   d_atoms_offset_temp.begin());
  //
  // _device_data->_d_atoms_offset.resize(d_atoms_offset_temp.size() + 1);
  // _device_data->_d_atoms_offset[0] = 0;
  // thrust::exclusive_scan(d_atoms_offset_temp.begin(), d_atoms_offset_temp.end(),
  //   _device_data->_d_atoms_offset.begin() + 1);


  /// angle
  thrust::copy(sd->_h_angle_type, sd->_h_angle_type + num_angles,
               _device_data->_d_angle_type.begin());
  thrust::copy(sd->_h_angle_id0, sd->_h_angle_id0 + num_angles,
               _device_data->_d_angle_id0.begin());
  thrust::copy(sd->_h_angle_id1, sd->_h_angle_id1 + num_angles,
               _device_data->_d_angle_id1.begin());
  thrust::copy(sd->_h_angle_id2, sd->_h_angle_id2 + num_angles,
               _device_data->_d_angle_id2.begin());
  thrust::copy(sd->_h_angle_id_vec, sd->_h_angle_id_vec + num_angles,
               _device_data->_d_angle_id_vec.begin());
  /// dihedral
  thrust::copy(sd->_h_dihedral_type, sd->_h_dihedral_type + num_dihedrals,
               _device_data->_d_dihedral_type.begin());
  thrust::copy(sd->_h_dihedral_id0, sd->_h_dihedral_id0 + num_dihedrals,
               _device_data->_d_dihedral_id0.begin());
  thrust::copy(sd->_h_dihedral_id1, sd->_h_dihedral_id1 + num_dihedrals,
               _device_data->_d_dihedral_id1.begin());
  thrust::copy(sd->_h_dihedral_id2, sd->_h_dihedral_id2 + num_dihedrals,
               _device_data->_d_dihedral_id2.begin());
  thrust::copy(sd->_h_dihedral_id3, sd->_h_dihedral_id3 + num_dihedrals,
               _device_data->_d_dihedral_id3.begin());

  //improper
  thrust::copy(sd->_h_improper_type, sd->_h_improper_type + num_impropers,
             _device_data->_d_improper_type.begin());
  thrust::copy(sd->_h_improper_id0, sd->_h_improper_id0 + num_impropers,
               _device_data->_d_improper_id0.begin());
  thrust::copy(sd->_h_improper_id1, sd->_h_improper_id1 + num_impropers,
               _device_data->_d_improper_id1.begin());
  thrust::copy(sd->_h_improper_id2, sd->_h_improper_id2 + num_impropers,
               _device_data->_d_improper_id2.begin());
  thrust::copy(sd->_h_improper_id3, sd->_h_improper_id3 + num_impropers,
               _device_data->_d_improper_id3.begin());

  /// (2) copy force field
  /// mass
 ///  _device_data->_d_mass.resize(num_atoms_type);
 ///  thrust::copy(fd->_h_mass, fd->_h_mass + num_atoms_type,
 ///               _device_data->_d_mass.begin());
  /// eps
 _device_data->_d_eps.resize(num_atoms_type);
  thrust::copy(fd->_h_eps, fd->_h_eps + num_atoms_type,
               _device_data->_d_eps.begin());
  /// sigma
  _device_data->_d_sigma.resize(num_atoms_type);
  thrust::copy(fd->_h_sigma, fd->_h_sigma + num_atoms_type,
               _device_data->_d_sigma.begin());
  /// bond
  _device_data->_d_bond_coeffs_k.resize(num_bonds_type);
  _device_data->_d_bond_coeffs_equilibrium.resize(num_bonds_type);
  ValidateBondCoeffArrayOrThrow("before_cvff_h2d_bond_upload",
                                fd->_h_bond_coeffs_equilibrium,
                                static_cast<std::size_t>(num_bonds_type));
  thrust::copy(fd->_h_bond_coeffs_k, fd->_h_bond_coeffs_k + num_bonds_type,
               _device_data->_d_bond_coeffs_k.begin());
  thrust::copy(fd->_h_bond_coeffs_equilibrium,
               fd->_h_bond_coeffs_equilibrium + num_bonds_type,
               _device_data->_d_bond_coeffs_equilibrium.begin());
  ValidateBondCoeffArrayOrThrow("after_cvff_h2d_bond_upload",
                                fd->_h_bond_coeffs_equilibrium,
                                static_cast<std::size_t>(num_bonds_type));
  /// angle
  _device_data->_d_angle_coeffs_k.resize(num_angles_type);
  _device_data->_d_angle_coeffs_equilibrium.resize(num_angles_type);
  thrust::copy(fd->_h_angle_coeffs_k, fd->_h_angle_coeffs_k + num_angles_type,
               _device_data->_d_angle_coeffs_k.begin());
  thrust::copy(fd->_h_angle_coeffs_equilibrium,
               fd->_h_angle_coeffs_equilibrium + num_angles_type,
               _device_data->_d_angle_coeffs_equilibrium.begin());
  //dihedral
  std::string dihedral_type = "null";

  if(*(_structure_info_data->_num_dihedrals)) {
    dihedral_type = DataManager::getInstance().getConfigData()->
      Get<std::string>("dihedral_type", "hyper_parameters", "force_field");
  }
  if (dihedral_type == "harmonic") {
    _device_data->_d_dihedral_coeffs_k.resize(num_dihedrals_type);
    _device_data->_d_dihedral_coeffs_sign.resize(num_dihedrals_type);
    _device_data->_d_dihedral_coeffs_multiplicity.resize(num_dihedrals_type);

    thrust::copy(fd->_h_dihedral_coeffs_k,
           fd->_h_dihedral_coeffs_k + num_dihedrals_type,
           _device_data->_d_dihedral_coeffs_k.begin());
    thrust::copy(fd->_h_dihedral_coeffs_sign,
                 fd->_h_dihedral_coeffs_sign + num_dihedrals_type,
                 _device_data->_d_dihedral_coeffs_sign.begin());
    thrust::copy(fd->_h_dihedral_coeffs_multiplicity,
                 fd->_h_dihedral_coeffs_multiplicity + num_dihedrals_type,
                 _device_data->_d_dihedral_coeffs_multiplicity.begin());
  }
  else if (dihedral_type == "opls") {
    _device_data->_d_dihedral_coeffs_k1.resize(num_dihedrals_type);
    _device_data->_d_dihedral_coeffs_k2.resize(num_dihedrals_type);
    _device_data->_d_dihedral_coeffs_k3.resize(num_dihedrals_type);
    _device_data->_d_dihedral_coeffs_k4.resize(num_dihedrals_type);

    thrust::copy(fd->_h_dihedral_coeffs_k1,
           fd->_h_dihedral_coeffs_k1 + num_dihedrals_type,
           _device_data->_d_dihedral_coeffs_k1.begin());
    thrust::copy(fd->_h_dihedral_coeffs_k2,
               fd->_h_dihedral_coeffs_k2 + num_dihedrals_type,
               _device_data->_d_dihedral_coeffs_k2.begin());
    thrust::copy(fd->_h_dihedral_coeffs_k3,
               fd->_h_dihedral_coeffs_k3 + num_dihedrals_type,
               _device_data->_d_dihedral_coeffs_k3.begin());
    thrust::copy(fd->_h_dihedral_coeffs_k4,
               fd->_h_dihedral_coeffs_k4 + num_dihedrals_type,
               _device_data->_d_dihedral_coeffs_k4.begin());

  }
  else if (dihedral_type == "fourier") {
    std::cout<< "test-thrust-DihedralFourier"<<std::endl;

    // First, we need to determine the total number of fourier terms across all types,
    // as this defines the size of the main coefficient arrays.
    size_t total_terms = 0;
    for (rbmd::Id i = 0; i < num_dihedrals_type; ++i) {
      total_terms += fd->_h_nterms[i];
    }
    std::cout<< "total_terms: "<<  total_terms <<std::endl;
    // Resize the device vectors to the appropriate sizes
    _device_data->_d_nterms.resize(num_dihedrals_type);
    _device_data->_d_fourier_offsets.resize(num_dihedrals_type);

    _device_data->_d_dihedral_coeffs_k.resize(total_terms);
    _device_data->_d_dihedral_coeffs_multiplicity.resize(total_terms);
    _device_data->_d_fourier_cos_shift.resize(total_terms);
    _device_data->_d_fourier_sin_shift.resize(total_terms);

    // Copy nterms and offsets arrays (size = num_dihedrals_type)
    thrust::copy(fd->_h_nterms,
                 fd->_h_nterms + num_dihedrals_type,
                 _device_data->_d_nterms.begin());

    thrust::copy(fd->_h_fourier_offsets,
                 fd->_h_fourier_offsets + num_dihedrals_type,
                 _device_data->_d_fourier_offsets.begin());

    // Copy the main flattened coefficient arrays (size = total_terms)
    thrust::copy(fd->_h_dihedral_coeffs_k,
                 fd->_h_dihedral_coeffs_k + total_terms,
                 _device_data->_d_dihedral_coeffs_k.begin());

    thrust::copy(fd->_h_dihedral_coeffs_multiplicity,
                 fd->_h_dihedral_coeffs_multiplicity + total_terms,
                 _device_data->_d_dihedral_coeffs_multiplicity.begin());

    thrust::copy(fd->_h_fourier_cos_shift,
                 fd->_h_fourier_cos_shift + total_terms,
                 _device_data->_d_fourier_cos_shift.begin());

    thrust::copy(fd->_h_fourier_sin_shift,
                 fd->_h_fourier_sin_shift + total_terms,
                 _device_data->_d_fourier_sin_shift.begin());
    std::cout<< "test-thrust-end-DihedralFourier"<<std::endl;
  }

  //improper
  std::string improper_type = "null";

  if(*(_structure_info_data->_num_impropers)) {
    improper_type = DataManager::getInstance().getConfigData()->
      Get<std::string>("improper_type", "hyper_parameters", "force_field");
  }
  if (improper_type == "harmonic") {
    _device_data->_d_improper_coeffs_k.resize(num_impropers_type);
    _device_data->_d_improper_coeffs_chi.resize(num_impropers_type);

    thrust::copy(fd->_h_improper_coeffs_k,
               fd->_h_improper_coeffs_k + num_impropers_type,
               _device_data->_d_improper_coeffs_k.begin());
    thrust::copy(fd->_h_improper_coeffs_degree,
                 fd->_h_improper_coeffs_degree + num_impropers_type,
                 _device_data->_d_improper_coeffs_chi.begin());
  }
  else if (improper_type == "cvff") {
    _device_data->_d_improper_coeffs_k.resize(num_impropers_type);
    _device_data->_d_improper_coeffs_d.resize(num_impropers_type);
    _device_data->_d_improper_coeffs_n.resize(num_impropers_type);

    thrust::copy(fd->_h_improper_coeffs_k,
           fd->_h_improper_coeffs_k + num_impropers_type,
           _device_data->_d_improper_coeffs_k.begin());
    thrust::copy(fd->_h_improper_coeffs_d,
               fd->_h_improper_coeffs_d + num_impropers_type,
               _device_data->_d_improper_coeffs_d.begin());
    thrust::copy(fd->_h_improper_coeffs_n,
               fd->_h_improper_coeffs_n + num_impropers_type,
               _device_data->_d_improper_coeffs_n.begin());
  }

  return true;
}

bool CVFFMemoryScheduler::asyncMemoryD2H() { return true; }
