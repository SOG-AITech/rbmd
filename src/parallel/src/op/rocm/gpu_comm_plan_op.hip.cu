#include "gpu_comm_plan_op.h"

#include "data_manager/include/model/box.h"

namespace op {
namespace {

constexpr int kMaxForwardFields = 12;

__device__ bool IsInside(const GhostPlanRegion& region, rbmd::Real x,
                         rbmd::Real y, rbmd::Real z) {
  return x >= region.low[0] && x < region.high[0] &&
         y >= region.low[1] && y < region.high[1] &&
         z >= region.low[2] && z < region.high[2];
}

__global__ void MakeGhostPlanKernel(
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const GhostPlanRegion* __restrict__ regions, rbmd::Id region_count,
    rbmd::Id atom_count, int* __restrict__ flags, int* __restrict__ owner,
    rbmd::Real* __restrict__ shift_x, rbmd::Real* __restrict__ shift_y,
    rbmd::Real* __restrict__ shift_z) {
  const rbmd::Id atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const rbmd::Real x = __ldg(&px[atom]);
  const rbmd::Real y = __ldg(&py[atom]);
  const rbmd::Real z = __ldg(&pz[atom]);
  int selected = -1;
  for (rbmd::Id region = 0; region < region_count; ++region) {
    if (IsInside(regions[region], x, y, z)) {
      selected = static_cast<int>(region);
      break;
    }
  }

  owner[atom] = selected;
  flags[atom] = selected >= 0 ? 1 : 0;
  if (selected >= 0) {
    shift_x[atom] = regions[selected].shift[0];
    shift_y[atom] = regions[selected].shift[1];
    shift_z[atom] = regions[selected].shift[2];
  } else {
    shift_x[atom] = rbmd::Real(0);
    shift_y[atom] = rbmd::Real(0);
    shift_z[atom] = rbmd::Real(0);
  }
}

__global__ void MarkLeavingPlanKernel(
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz,
    const GhostPlanRegion* __restrict__ regions, rbmd::Id region_count,
    rbmd::Id atom_count, const int* __restrict__ deferred_mask,
    int* __restrict__ flags, int* __restrict__ owner,
    rbmd::Real* __restrict__ shift_x, rbmd::Real* __restrict__ shift_y,
    rbmd::Real* __restrict__ shift_z) {
  const rbmd::Id atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  int selected = -1;
  if (deferred_mask == nullptr || deferred_mask[atom] == 0) {
    const rbmd::Real x = __ldg(&px[atom]);
    const rbmd::Real y = __ldg(&py[atom]);
    const rbmd::Real z = __ldg(&pz[atom]);
    for (rbmd::Id region = 0; region < region_count; ++region) {
      if (IsInside(regions[region], x, y, z)) {
        selected = static_cast<int>(region);
        break;
      }
    }
  }

  owner[atom] = selected;
  flags[atom] = selected >= 0 ? 1 : 0;
  if (selected >= 0) {
    shift_x[atom] = regions[selected].shift[0];
    shift_y[atom] = regions[selected].shift[1];
    shift_z[atom] = regions[selected].shift[2];
  } else {
    shift_x[atom] = rbmd::Real(0);
    shift_y[atom] = rbmd::Real(0);
    shift_z[atom] = rbmd::Real(0);
  }
}

__device__ __forceinline__ rbmd::Real ApplyLeavingShiftAndClamp(
    rbmd::Real position, rbmd::Real shift, rbmd::Real box_min,
    rbmd::Real box_max) {
  rbmd::Real shifted = position + shift;
  if (shift < 0 && shifted < box_min) {
    shifted = NEXTAFTER(box_min, box_min + rbmd::Real(1));
  } else if (shift > 0 && shifted >= box_max) {
    shifted = NEXTAFTER(box_max, box_max - rbmd::Real(1));
  }
  return shifted;
}

__global__ void PackLeavingSoAKernel(
    const int* __restrict__ indices, const rbmd::Real* __restrict__ shift_x,
    const rbmd::Real* __restrict__ shift_y,
    const rbmd::Real* __restrict__ shift_z, rbmd::Id count,
    const rbmd::Id* __restrict__ atom_ids,
    const rbmd::Id* __restrict__ atom_types,
    const rbmd::Id* __restrict__ molecule_ids,
    const rbmd::Real* __restrict__ px, const rbmd::Real* __restrict__ py,
    const rbmd::Real* __restrict__ pz, const rbmd::Real* __restrict__ vx,
    const rbmd::Real* __restrict__ vy, const rbmd::Real* __restrict__ vz,
    const rbmd::Real* __restrict__ charge, const rbmd::Id* image_x,
    const rbmd::Id* image_y, const rbmd::Id* image_z,
    bool has_molecule, bool has_charge,
    char* buffer, rbmd::Real box_min_x, rbmd::Real box_min_y,
    rbmd::Real box_min_z, rbmd::Real box_max_x, rbmd::Real box_max_y,
    rbmd::Real box_max_z) {
  const rbmd::Id out = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= count) {
    return;
  }

  const int source = __ldg(&indices[out]);
  auto* packed_ids = reinterpret_cast<rbmd::Id*>(buffer);
  auto* packed_types = packed_ids + count;
  auto* packed_molecule = has_molecule ? packed_types + count : nullptr;
  char* real_buffer = reinterpret_cast<char*>(packed_types + count);
  if (has_molecule) {
    real_buffer = reinterpret_cast<char*>(packed_molecule + count);
  }
  auto* packed_image_x = reinterpret_cast<rbmd::Id*>(real_buffer);
  auto* packed_image_y = packed_image_x + count;
  auto* packed_image_z = packed_image_y + count;
  const rbmd::Id padding =
      !has_molecule && sizeof(rbmd::Real) > sizeof(rbmd::Id) ? count : 0;
  auto* packed_x = reinterpret_cast<rbmd::Real*>(packed_image_z + count + padding);
  auto* packed_y = packed_x + count;
  auto* packed_z = packed_y + count;
  auto* packed_vx = packed_z + count;
  auto* packed_vy = packed_vx + count;
  auto* packed_vz = packed_vy + count;
  auto* packed_charge = packed_vz + count;
  auto* packed_ux = packed_charge + (has_charge ? count : 0);
  auto* packed_uy = packed_ux + count;
  auto* packed_uz = packed_uy + count;

  packed_ids[out] = __ldg(&atom_ids[source]);
  packed_types[out] = __ldg(&atom_types[source]);
  if (has_molecule) {
    packed_molecule[out] = __ldg(&molecule_ids[source]);
  }
  packed_x[out] = ApplyLeavingShiftAndClamp(
      __ldg(&px[source]), __ldg(&shift_x[source]), box_min_x, box_max_x);
  packed_y[out] = ApplyLeavingShiftAndClamp(
      __ldg(&py[source]), __ldg(&shift_y[source]), box_min_y, box_max_y);
  packed_z[out] = ApplyLeavingShiftAndClamp(
      __ldg(&pz[source]), __ldg(&shift_z[source]), box_min_z, box_max_z);
  // A periodic coordinate shift must leave the physical unwrapped position unchanged.
  const rbmd::Real lx = box_max_x - box_min_x;
  const rbmd::Real ly = box_max_y - box_min_y;
  const rbmd::Real lz = box_max_z - box_min_z;
  packed_image_x[out] = image_x[source] - static_cast<rbmd::Id>(ROUND(shift_x[source] / lx));
  packed_image_y[out] = image_y[source] - static_cast<rbmd::Id>(ROUND(shift_y[source] / ly));
  packed_image_z[out] = image_z[source] - static_cast<rbmd::Id>(ROUND(shift_z[source] / lz));
  packed_ux[out] = packed_x[out] + packed_image_x[out] * lx;
  packed_uy[out] = packed_y[out] + packed_image_y[out] * ly;
  packed_uz[out] = packed_z[out] + packed_image_z[out] * lz;
  packed_vx[out] = __ldg(&vx[source]);
  packed_vy[out] = __ldg(&vy[source]);
  packed_vz[out] = __ldg(&vz[source]);
  if (has_charge) {
    packed_charge[out] = __ldg(&charge[source]);
  }
}

__global__ void PackGhostKernel(
    const rbmd::Id* __restrict__ indices,
    const rbmd::Real* __restrict__ shift_x,
    const rbmd::Real* __restrict__ shift_y,
    const rbmd::Real* __restrict__ shift_z, rbmd::Id count,
    const rbmd::Id* __restrict__ atom_ids,
    const rbmd::Id* __restrict__ atom_types, const rbmd::Real* __restrict__ px,
    const rbmd::Real* __restrict__ py, const rbmd::Real* __restrict__ pz,
    const rbmd::Real* __restrict__ charge, bool has_charge, char* buffer) {
  const rbmd::Id out = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= count) {
    return;
  }
  const rbmd::Id source = __ldg(&indices[out]);
  auto* ids = reinterpret_cast<rbmd::Id*>(buffer);
  auto* types = reinterpret_cast<rbmd::Id*>(buffer + count * sizeof(rbmd::Id));
  auto* packed_x = reinterpret_cast<rbmd::Real*>(
      buffer + count * sizeof(rbmd::Id) * 2);
  auto* packed_y = packed_x + count;
  auto* packed_z = packed_y + count;
  auto* packed_charge = packed_z + count;
  ids[out] = __ldg(&atom_ids[source]);
  types[out] = __ldg(&atom_types[source]);
  packed_x[out] = __ldg(&px[source]) + __ldg(&shift_x[out]);
  packed_y[out] = __ldg(&py[source]) + __ldg(&shift_y[out]);
  packed_z[out] = __ldg(&pz[source]) + __ldg(&shift_z[out]);
  if (has_charge) {
    packed_charge[out] = __ldg(&charge[source]);
  }
}

__global__ void UnpackGhostKernel(
    const char* buffer, rbmd::Id count, rbmd::Id destination_offset,
    rbmd::Id* __restrict__ atom_ids, rbmd::Id* __restrict__ atom_types,
    rbmd::Real* __restrict__ px, rbmd::Real* __restrict__ py,
    rbmd::Real* __restrict__ pz, rbmd::Real* __restrict__ charge,
    bool has_charge) {
  const rbmd::Id input = blockIdx.x * blockDim.x + threadIdx.x;
  if (input >= count) {
    return;
  }
  const auto* ids = reinterpret_cast<const rbmd::Id*>(buffer);
  const auto* types =
      reinterpret_cast<const rbmd::Id*>(buffer + count * sizeof(rbmd::Id));
  const auto* packed_x = reinterpret_cast<const rbmd::Real*>(
      buffer + count * sizeof(rbmd::Id) * 2);
  const auto* packed_y = packed_x + count;
  const auto* packed_z = packed_y + count;
  const auto* packed_charge = packed_z + count;
  const rbmd::Id output = destination_offset + input;
  atom_ids[output] = ids[input];
  atom_types[output] = types[input];
  px[output] = packed_x[input];
  py[output] = packed_y[input];
  pz[output] = packed_z[input];
  if (has_charge) {
    charge[output] = packed_charge[input];
  }
}

__global__ void PackForwardFieldsKernel(
    const rbmd::Id* __restrict__ indices,
    const rbmd::Real* __restrict__ shift_x,
    const rbmd::Real* __restrict__ shift_y,
    const rbmd::Real* __restrict__ shift_z,
    const rbmd::Real* const* __restrict__ source,
    const int* __restrict__ shift_dimension, int field_count, rbmd::Id count,
    rbmd::Real* __restrict__ packed_values) {
  const rbmd::Id out = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= count) {
    return;
  }
  const rbmd::Id src = __ldg(&indices[out]);
  for (int field = 0; field < field_count; ++field) {
    rbmd::Real value = source[field][src];
    if (shift_dimension[field] == 0) {
      value += shift_x[out];
    } else if (shift_dimension[field] == 1) {
      value += shift_y[out];
    } else if (shift_dimension[field] == 2) {
      value += shift_z[out];
    }
    packed_values[static_cast<rbmd::Id>(field) * count + out] = value;
  }
}

__global__ void ApplyForwardFieldsKernel(
    const rbmd::Real* __restrict__ packed_values,
    const rbmd::Id* __restrict__ indices,
    rbmd::Real* const* __restrict__ destination, int field_count,
    rbmd::Id count) {
  const rbmd::Id input = blockIdx.x * blockDim.x + threadIdx.x;
  if (input >= count) {
    return;
  }
  const rbmd::Id dst = __ldg(&indices[input]);
  for (int field = 0; field < field_count; ++field) {
    destination[field][dst] =
        packed_values[static_cast<rbmd::Id>(field) * count + input];
  }
}

unsigned int Blocks(rbmd::Id count) {
  return static_cast<unsigned int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

}  // namespace

void MakeGhostPlanOp<device::DEVICE_GPU>::operator()(
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const GhostPlanRegion* regions, rbmd::Id region_count,
    rbmd::Id atom_count, int* flags, int* owner, rbmd::Real* shift_x,
    rbmd::Real* shift_y, rbmd::Real* shift_z, STREAM_T stream) {
  if (atom_count <= 0) {
    return;
  }
  CHECK_KERNEL(
      MakeGhostPlanKernel<<<Blocks(atom_count), BLOCK_SIZE, 0, stream>>>(
          px, py, pz, regions, region_count, atom_count, flags, owner, shift_x,
          shift_y, shift_z));
}

void MakeLeavingPlanOp<device::DEVICE_GPU>::operator()(
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const GhostPlanRegion* regions, rbmd::Id region_count, rbmd::Id atom_count,
    const int* deferred_mask, int* flags, int* owner, rbmd::Real* shift_x,
    rbmd::Real* shift_y, rbmd::Real* shift_z, STREAM_T stream) {
  if (atom_count <= 0) {
    return;
  }
  CHECK_KERNEL(
      MarkLeavingPlanKernel<<<Blocks(atom_count), BLOCK_SIZE, 0, stream>>>(
          px, py, pz, regions, region_count, atom_count, deferred_mask, flags,
          owner, shift_x, shift_y, shift_z));
}

void PackLeavingSoAOp<device::DEVICE_GPU>::operator()(
    const int* indices, const rbmd::Real* shift_x, const rbmd::Real* shift_y,
    const rbmd::Real* shift_z, rbmd::Id count, const rbmd::Id* atom_ids,
    const rbmd::Id* atom_types, const rbmd::Id* molecule_ids,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* vx, const rbmd::Real* vy, const rbmd::Real* vz,
    const rbmd::Real* charge, const rbmd::Id* image_x,
    const rbmd::Id* image_y, const rbmd::Id* image_z,
    bool has_molecule, bool has_charge, char* buffer,
    const Box& box, STREAM_T stream) {
  if (count <= 0) {
    return;
  }
  CHECK_KERNEL(PackLeavingSoAKernel<<<Blocks(count), BLOCK_SIZE, 0, stream>>>(
      indices, shift_x, shift_y, shift_z, count, atom_ids, atom_types,
      molecule_ids, px, py, pz, vx, vy, vz, charge, image_x, image_y, image_z,
      has_molecule, has_charge,
      buffer, box._coord_min[0], box._coord_min[1], box._coord_min[2],
      box._coord_max[0], box._coord_max[1], box._coord_max[2]));
}

void PackGhostOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id* indices, const rbmd::Real* shift_x,
    const rbmd::Real* shift_y, const rbmd::Real* shift_z, rbmd::Id count,
    const rbmd::Id* atom_ids, const rbmd::Id* atom_types,
    const rbmd::Real* px, const rbmd::Real* py, const rbmd::Real* pz,
    const rbmd::Real* charge, bool has_charge, char* buffer, STREAM_T stream) {
  if (count <= 0) {
    return;
  }
  CHECK_KERNEL(PackGhostKernel<<<Blocks(count), BLOCK_SIZE, 0, stream>>>(
      indices, shift_x, shift_y, shift_z, count, atom_ids, atom_types, px, py,
      pz, charge, has_charge, buffer));
}

void UnpackGhostOp<device::DEVICE_GPU>::operator()(
    const char* buffer, rbmd::Id count, rbmd::Id destination_offset,
    rbmd::Id* atom_ids, rbmd::Id* atom_types, rbmd::Real* px, rbmd::Real* py,
    rbmd::Real* pz, rbmd::Real* charge, bool has_charge, STREAM_T stream) {
  if (count <= 0) {
    return;
  }
  CHECK_KERNEL(UnpackGhostKernel<<<Blocks(count), BLOCK_SIZE, 0, stream>>>(
      buffer, count, destination_offset, atom_ids, atom_types, px, py, pz,
      charge, has_charge));
}

void PackForwardFieldsOp<device::DEVICE_GPU>::operator()(
    const rbmd::Id* indices, const rbmd::Real* shift_x,
    const rbmd::Real* shift_y, const rbmd::Real* shift_z,
    const rbmd::Real* const* source, const int* shift_dimension,
    int field_count, rbmd::Id count, rbmd::Real* packed_values,
    STREAM_T stream) {
  if (count <= 0 || field_count <= 0 || field_count > kMaxForwardFields) {
    return;
  }
  CHECK_KERNEL(PackForwardFieldsKernel<<<Blocks(count), BLOCK_SIZE, 0, stream>>>(
      indices, shift_x, shift_y, shift_z, source, shift_dimension, field_count,
      count, packed_values));
}

void ApplyForwardFieldsOp<device::DEVICE_GPU>::operator()(
    const rbmd::Real* packed_values, const rbmd::Id* indices,
    rbmd::Real* const* destination, int field_count, rbmd::Id count,
    STREAM_T stream) {
  if (count <= 0 || field_count <= 0 || field_count > kMaxForwardFields) {
    return;
  }
  CHECK_KERNEL(ApplyForwardFieldsKernel<<<Blocks(count), BLOCK_SIZE, 0, stream>>>(
      packed_values, indices, destination, field_count, count));
}

}  // namespace op
