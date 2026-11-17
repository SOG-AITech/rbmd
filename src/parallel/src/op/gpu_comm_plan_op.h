#pragma once

#include "common/device_types.h"
#include "common/rbmd_define.h"
#include "common/types.h"

class Box;

namespace op {

struct GhostPlanRegion {
  rbmd::Real low[3]{};
  rbmd::Real high[3]{};
  rbmd::Real shift[3]{};
};

template <typename DEVICE>
struct MakeGhostPlanOp {
  void operator()(const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const GhostPlanRegion* regions,
                  rbmd::Id region_count, rbmd::Id atom_count, int* flags,
                  int* owner, rbmd::Real* shift_x, rbmd::Real* shift_y,
                  rbmd::Real* shift_z, STREAM_T stream = nullptr);
};

template <typename DEVICE>
struct MakeLeavingPlanOp {
  void operator()(const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const GhostPlanRegion* regions,
                  rbmd::Id region_count, rbmd::Id atom_count,
                  const int* deferred_mask, int* flags, int* owner,
                  rbmd::Real* shift_x, rbmd::Real* shift_y, rbmd::Real* shift_z,
                  STREAM_T stream = nullptr);
};

template <typename DEVICE>
struct PackLeavingSoAOp {
  void operator()(const int* indices, const rbmd::Real* shift_x,
                  const rbmd::Real* shift_y, const rbmd::Real* shift_z,
                  rbmd::Id count, const rbmd::Id* atom_ids,
                  const rbmd::Id* atom_types, const rbmd::Id* molecule_ids,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* vx,
                  const rbmd::Real* vy, const rbmd::Real* vz,
                  const rbmd::Real* charge, const rbmd::Id* image_x,
                  const rbmd::Id* image_y, const rbmd::Id* image_z,
                  bool has_molecule, bool has_charge,
                  char* buffer, const Box& box, STREAM_T stream = nullptr);
};

template <typename DEVICE>
struct PackGhostOp {
  void operator()(const rbmd::Id* indices, const rbmd::Real* shift_x,
                  const rbmd::Real* shift_y, const rbmd::Real* shift_z,
                  rbmd::Id count, const rbmd::Id* atom_ids,
                  const rbmd::Id* atom_types, const rbmd::Real* px,
                  const rbmd::Real* py, const rbmd::Real* pz,
                  const rbmd::Real* charge, bool has_charge, char* buffer,
                  STREAM_T stream = nullptr);
};

template <typename DEVICE>
struct UnpackGhostOp {
  void operator()(const char* buffer, rbmd::Id count,
                  rbmd::Id destination_offset, rbmd::Id* atom_ids,
                  rbmd::Id* atom_types, rbmd::Real* px, rbmd::Real* py,
                  rbmd::Real* pz, rbmd::Real* charge, bool has_charge,
                  STREAM_T stream = nullptr);
};

template <typename DEVICE>
struct PackForwardFieldsOp {
  void operator()(const rbmd::Id* indices, const rbmd::Real* shift_x,
                  const rbmd::Real* shift_y, const rbmd::Real* shift_z,
                  const rbmd::Real* const* source, const int* shift_dimension,
                  int field_count, rbmd::Id count, rbmd::Real* packed_values,
                  STREAM_T stream = nullptr);
};

template <typename DEVICE>
struct ApplyForwardFieldsOp {
  void operator()(const rbmd::Real* packed_values, const rbmd::Id* indices,
                  rbmd::Real* const* destination, int field_count,
                  rbmd::Id count, STREAM_T stream = nullptr);
};

template <>
struct MakeGhostPlanOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const GhostPlanRegion* regions,
                  rbmd::Id region_count, rbmd::Id atom_count, int* flags,
                  int* owner, rbmd::Real* shift_x, rbmd::Real* shift_y,
                  rbmd::Real* shift_z, STREAM_T stream = nullptr);
};

template <>
struct MakeLeavingPlanOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const GhostPlanRegion* regions,
                  rbmd::Id region_count, rbmd::Id atom_count,
                  const int* deferred_mask, int* flags, int* owner,
                  rbmd::Real* shift_x, rbmd::Real* shift_y, rbmd::Real* shift_z,
                  STREAM_T stream = nullptr);
};

template <>
struct PackLeavingSoAOp<device::DEVICE_GPU> {
  void operator()(const int* indices, const rbmd::Real* shift_x,
                  const rbmd::Real* shift_y, const rbmd::Real* shift_z,
                  rbmd::Id count, const rbmd::Id* atom_ids,
                  const rbmd::Id* atom_types, const rbmd::Id* molecule_ids,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* vx,
                  const rbmd::Real* vy, const rbmd::Real* vz,
                  const rbmd::Real* charge, const rbmd::Id* image_x,
                  const rbmd::Id* image_y, const rbmd::Id* image_z,
                  bool has_molecule, bool has_charge,
                  char* buffer, const Box& box, STREAM_T stream = nullptr);
};

template <>
struct PackGhostOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Id* indices, const rbmd::Real* shift_x,
                  const rbmd::Real* shift_y, const rbmd::Real* shift_z,
                  rbmd::Id count, const rbmd::Id* atom_ids,
                  const rbmd::Id* atom_types, const rbmd::Real* px,
                  const rbmd::Real* py, const rbmd::Real* pz,
                  const rbmd::Real* charge, bool has_charge, char* buffer,
                  STREAM_T stream = nullptr);
};

template <>
struct UnpackGhostOp<device::DEVICE_GPU> {
  void operator()(const char* buffer, rbmd::Id count,
                  rbmd::Id destination_offset, rbmd::Id* atom_ids,
                  rbmd::Id* atom_types, rbmd::Real* px, rbmd::Real* py,
                  rbmd::Real* pz, rbmd::Real* charge, bool has_charge,
                  STREAM_T stream = nullptr);
};

template <>
struct PackForwardFieldsOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Id* indices, const rbmd::Real* shift_x,
                  const rbmd::Real* shift_y, const rbmd::Real* shift_z,
                  const rbmd::Real* const* source, const int* shift_dimension,
                  int field_count, rbmd::Id count, rbmd::Real* packed_values,
                  STREAM_T stream = nullptr);
};

template <>
struct ApplyForwardFieldsOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Real* packed_values, const rbmd::Id* indices,
                  rbmd::Real* const* destination, int field_count,
                  rbmd::Id count, STREAM_T stream = nullptr);
};

}  // namespace op
