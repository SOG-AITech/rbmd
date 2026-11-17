#pragma once
#include <cstdint>
#include <limits>

namespace rbmd {
#if USE_DOUBLE
using Real = double;
#else
using Real = float;
#endif

#if USE_64BIT_IDS
using Id = long long;
using Int = long long;
#else
using Id = int;
using Int = int;
#endif
// Distinguish global identity/counts from bounded local array positions.
using GlobalAtomId = long long;
using GlobalCount = long long;
using NeighborOffset = long long;
using LocalAtomIndex = std::int32_t;
using Energy = double;
static_assert(sizeof(GlobalAtomId) == 8 && sizeof(NeighborOffset) == 8);
static_assert(sizeof(Energy) == 8);
#if USE_64BIT_IDS
static_assert(sizeof(Id) == 8, "64-bit ID configuration did not take effect");
#endif
#if USE_DOUBLE
static_assert(sizeof(Real) == 8, "Double precision configuration did not take effect");
#endif
}  // namespace rbmd
