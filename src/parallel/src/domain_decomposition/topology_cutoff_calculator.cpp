#include "domain_decomposition/topology_cutoff_calculator.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "data_manager.h"
#include "force_field/cvff_force_field_data.h"

namespace {

std::string FormatBondCoeffSample(const rbmd::Real* values, rbmd::Id count,
                                  rbmd::Id limit = 8) {
  if (values == nullptr || count <= 0) {
    return "[]";
  }

  std::ostringstream oss;
  oss << '[';
  const rbmd::Id sample_count = std::min(count, limit);
  for (rbmd::Id i = 0; i < sample_count; ++i) {
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

}  // namespace

double TopologyCutoffCalculator::Calculate(double neighbor_skin) {
  // 从 DataManager 获取系统数据
  auto md_data = DataManager::getInstance().getMDData();
  if (!md_data || !md_data->_structure_info_data) {
    // 如果数据不可用，返回零截断
    return 0.0;
  }

  auto structure_info = md_data->_structure_info_data;

  // 判断系统中是否有各种拓扑类型
  bool has_bonds =
      (structure_info->_num_bonds && *structure_info->_num_bonds > 0);
  bool has_angles =
      (structure_info->_num_angles && *structure_info->_num_angles > 0);
  bool has_dihedrals =
      (structure_info->_num_dihedrals && *structure_info->_num_dihedrals > 0);
  bool has_impropers =
      (structure_info->_num_impropers && *structure_info->_num_impropers > 0);

  // 获取最大平衡键长
  double max_bond = GetMaxEquilibriumBondLength();

  // Bond: 1.5 * max_bond + skin
  double bond_cutoff = has_bonds ? (1.5 * max_bond + neighbor_skin) : 0.0;

  // Angle: 2.0 * max_bond + skin
  double angle_cutoff = has_angles ? (2.0 * max_bond + neighbor_skin) : 0.0;

  // Dihedral/Improper: 3.0 * max_bond + skin
  double dihedral_cutoff =
      (has_dihedrals || has_impropers) ? (3.0 * max_bond + neighbor_skin) : 0.0;

  // 返回最大截断距离
  return std::max({bond_cutoff, angle_cutoff, dihedral_cutoff});
}

double TopologyCutoffCalculator::GetMaxEquilibriumBondLength() {
  auto md_data = DataManager::getInstance().getMDData();
  if (!md_data || !md_data->_structure_info_data || !md_data->_force_field_data) {
    return 1.8;
  }

  auto* cvff_force_field =
      dynamic_cast<CVFFForceFieldData*>(md_data->_force_field_data.get());
  auto* structure_info = md_data->_structure_info_data.get();
  if (!cvff_force_field || !structure_info->_num_bonds_type ||
      *structure_info->_num_bonds_type <= 0 ||
      cvff_force_field->_h_bond_coeffs_equilibrium == nullptr) {
    return 1.8;
  }

  double max_bond_length = 0.0;
  const auto bond_type_count = *structure_info->_num_bonds_type;
  rbmd::Id max_bond_index = -1;
  for (rbmd::Id i = 0; i < bond_type_count; ++i) {
    const double value = cvff_force_field->_h_bond_coeffs_equilibrium[i];
    if (!std::isfinite(value)) {
      std::ostringstream oss;
      oss << "TopologyCutoffCalculator observed non-finite bond equilibrium at index "
          << i << " / " << bond_type_count
          << ", coeff_ptr="
          << static_cast<const void*>(cvff_force_field->_h_bond_coeffs_equilibrium)
          << ", sample="
          << FormatBondCoeffSample(cvff_force_field->_h_bond_coeffs_equilibrium,
                                   bond_type_count);
      throw std::runtime_error(oss.str());
    }
    if (value > max_bond_length) {
      max_bond_length = value;
      max_bond_index = i;
    }
  }

  if (max_bond_length > 1000.0) {
    std::ostringstream oss;
    oss << "TopologyCutoffCalculator observed suspicious max bond equilibrium="
        << max_bond_length << " at index " << max_bond_index
        << " / " << bond_type_count
        << ", coeff_ptr="
        << static_cast<const void*>(cvff_force_field->_h_bond_coeffs_equilibrium)
        << ", sample="
        << FormatBondCoeffSample(cvff_force_field->_h_bond_coeffs_equilibrium,
                                 bond_type_count);
    throw std::runtime_error(oss.str());
  }

  if (max_bond_length > 0.0) {
    return max_bond_length;
  }

  // 如果数据不可用或为0，使用保守的默认值（考虑键振动）
  return 1.8;
}
