#pragma once

#include "common/device_types.h"
#include "common/types.h"
#include "model/box.h"

namespace op {

template <typename DEVICE>
struct CacheCheckOp {
  void operator()(const rbmd::Id* atom_ids, const rbmd::Id* cached_atom_ids,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* cached_px,
                  const rbmd::Real* cached_py, const rbmd::Real* cached_pz,
                  Box box, rbmd::Id atom_count,
                  rbmd::Real rebuild_threshold_sq, bool collect_max_displacement,
                  rbmd::Real* status);
};

template <>
struct CacheCheckOp<device::DEVICE_GPU> {
  void operator()(const rbmd::Id* atom_ids, const rbmd::Id* cached_atom_ids,
                  const rbmd::Real* px, const rbmd::Real* py,
                  const rbmd::Real* pz, const rbmd::Real* cached_px,
                  const rbmd::Real* cached_py, const rbmd::Real* cached_pz,
                  Box box, rbmd::Id atom_count,
                  rbmd::Real rebuild_threshold_sq, bool collect_max_displacement,
                  rbmd::Real* status);
};

}  // namespace op
