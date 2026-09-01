#pragma once
#include "common/rbmd_define.h"
struct HaloRegion {
  rbmd::Real rmin[3];  // lower corner
  rbmd::Real rmax[3];  // higher corner
  int offset[3];       // offset (direction) of the halo region
  rbmd::Real width;    // Halo width (e.g. one cutoff)
};
