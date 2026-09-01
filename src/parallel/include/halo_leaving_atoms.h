#pragma once
#include "common/rbmd_define.h"
// Define a struct for the Halo particles  //TODO 键可能需要ID的
struct HaloAtom {
  rbmd::Real rx;
  rbmd::Real ry;
  rbmd::Real rz;
};

// Define a struct for the Leaving particles
struct LeavingAtom {
  rbmd::Id id;
  rbmd::Real rx;
  rbmd::Real ry;
  rbmd::Real rz;
  rbmd::Real vx;
  rbmd::Real vy;
  rbmd::Real vz;
  rbmd::Real charge;
};
