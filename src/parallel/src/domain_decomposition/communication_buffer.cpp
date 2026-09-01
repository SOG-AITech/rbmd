#include "communication_buffer.h"

#include <stdexcept>
#include <string>

#include "data_manager.h"
#include "domain_decomposition/topology_pack_op.h"

CommunicationBuffer::CommunicationBuffer() {
  auto atom_style =
      DataManager::getInstance().getConfigData()->Get<std::string>(
          "atom_style", "init_configuration", "read_data");
  
  // ========== 旧模式：基础 buffer 大小（兼容性保留） ==========
  if ("atomic" == atom_style) {
    this->_leaving_size = 2 * sizeof(rbmd::Id) + 6 * sizeof(rbmd::Real);
    this->_ghost_size = 2 * sizeof(rbmd::Id) + 3 * sizeof(rbmd::Real);
  } else if ("charge" == atom_style) {
    this->_leaving_size = 2 * sizeof(rbmd::Id) + 7 * sizeof(rbmd::Real);
    this->_ghost_size = 2 * sizeof(rbmd::Id) + 4 * sizeof(rbmd::Real);
  } else if ("full" == atom_style) {
    this->_leaving_size = 3 * sizeof(rbmd::Id) + 7 * sizeof(rbmd::Real);
    this->_ghost_size = 2 * sizeof(rbmd::Id) + 4 * sizeof(rbmd::Real);
  } else {
    throw std::runtime_error("Error atoms style");
  }
  
  // 只有 full atom_style 需要 topology exchange；charge/atomic 保持 ref 的旧 SoA 布局。
  if ("full" == atom_style) {
    auto& device_data = DataManager::getInstance().getDeviceData();
    _bond_per_atom = device_data->bond_per_atom;
    _angle_per_atom = device_data->angle_per_atom;
    _dihedral_per_atom = device_data->dihedral_per_atom;
    _improper_per_atom = device_data->improper_per_atom;
    _maxspecial = device_data->maxspecial;

    _size_exchange = op::ComputeSizeExchange(
        _bond_per_atom, _angle_per_atom, _dihedral_per_atom,
        _improper_per_atom, _maxspecial);
  }
}

void CommunicationBuffer::ResizeLeaving(unsigned int leaving_size) {
  // 旧模式：id  type molecular_id px py pz vx vy vz  charge
  this->_leaving_atoms.resize(this->_leaving_size * leaving_size);
}

void CommunicationBuffer::ResizeGhost(unsigned int halo_size) {
  // id type px py pz [charge]
  this->_ghost_atoms.resize(this->_ghost_size * halo_size);
}

void CommunicationBuffer::ResizeLeavingByByteSize(unsigned long leaving_size) {
  this->_leaving_atoms.resize(leaving_size);
}

void CommunicationBuffer::ResizeGhostByByteSize(unsigned long halo_size) {
  this->_ghost_atoms.resize(halo_size);
}

void CommunicationBuffer::ResizeShakeForwardByByteSize(unsigned long byte_size) {
  _shake_forward_atoms.resize(byte_size);
  _shake_forward_size = byte_size;
}

void CommunicationBuffer::ResizeShakeReverseByByteSize(unsigned long byte_size) {
  _shake_reverse_atoms.resize(byte_size);
  _shake_reverse_size = byte_size;
}

void CommunicationBuffer::ResizeLeavingExchange(unsigned int num_atoms, int size_exchange) {
  // LAMMPS layout：double buffer
  // 总大小 = num_atoms * size_exchange (单位：double)
  // 确保 buffer 对齐（double 本身已对齐）
  const size_t total_doubles = static_cast<size_t>(num_atoms) * static_cast<size_t>(size_exchange);
  this->_leaving_atoms_double.resize(total_doubles);
}

void CommunicationBuffer::Clear() {
  //! clear会删除元素释放空间
  this->_leaving_atoms.resize(0);
  this->_leaving_atoms_double.resize(0);
  this->_ghost_atoms.resize(0);
  this->_shake_forward_atoms.resize(0);
  this->_shake_reverse_atoms.resize(0);
}
