#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/types.h"
#include "common/json.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RBMD_TRAJECTORY_HD __host__ __device__
#else
#define RBMD_TRAJECTORY_HD
#endif

namespace rbmd::trajectory {
// Bounds are absolute wrapped coordinates; each interval is [lo, hi).
// A window crossing a periodic boundary is represented by two regions.
struct Region {
  Real lo[3]{};
  Real hi[3]{};
  bool enabled[3]{};
  RBMD_TRAJECTORY_HD bool Contains(Real x, Real y, Real z) const {
    const Real p[3] = {x, y, z};
    for (int d = 0; d < 3; ++d)
      if (enabled[d] && !(p[d] >= lo[d] && p[d] < hi[d])) return false;
    return true;
  }
};

RBMD_TRAJECTORY_HD inline bool ContainsId(const Id* ids, size_t n, Id id) {
  size_t first = 0, last = n;
  while (first < last) {
    const size_t mid = first + (last - first) / 2;
    if (ids[mid] < id) first = mid + 1;
    else last = mid;
  }
  return first < n && ids[first] == id;
}

// Device pointers in production, ordinary arrays in host tests. Id/type filters
// use internal zero-based integer values, never floating-point conversions.
struct Predicate {
  const Id* atom_ids{};
  const Id* atom_types{};
  const Real* x{};
  const Real* y{};
  const Real* z{};
  const Id* ids{};
  size_t num_ids{};
  bool filter_ids{false};  // true with zero ids deliberately selects nothing
  const Id* types{};
  size_t num_types{};
  const Region* regions{};
  size_t num_regions{};

  RBMD_TRAJECTORY_HD bool operator()(size_t i) const {
    if (filter_ids && !ContainsId(ids, num_ids, atom_ids[i])) return false;
    if (num_types && !ContainsId(types, num_types, atom_types[i])) return false;
    if (!num_regions) return true;
    for (size_t r = 0; r < num_regions; ++r)
      if (regions[r].Contains(x[i], y[i], z[i])) return true;
    return false;
  }
};

inline const std::vector<std::string>& AllFields() {
  static const std::vector<std::string> fields{
      "id", "type", "q", "x", "y", "z", "xu", "yu", "zu", "vx", "vy", "vz"};
  return fields;
}

struct Options {
  std::vector<std::string> fields{AllFields()};
  std::vector<Id> ids, types;
  std::vector<Region> regions;
  bool initial{false};
  bool filter_ids{false};
  // Global cap, checked on the GPU before allocating/copying atom fields.
  size_t max_atoms{200000};
};

inline std::vector<Id> ReadIds(const nlohmann::ordered_json& values) {
  if (!values.is_array() || values.empty())
    throw std::runtime_error("trajectory selection ids/types must be a nonempty array");
  std::vector<Id> result;
  for (const auto& value : values) {
    if (!value.is_number_integer() || value < 1 ||
        value > std::numeric_limits<Id>::max())
      throw std::runtime_error("trajectory selection ids/types must be positive representable integers");
    result.push_back(value.get<Id>() - 1);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

inline Region ReadRegion(const nlohmann::ordered_json& node) {
  if (!node.is_object() || node.empty())
    throw std::runtime_error("trajectory region must contain x, y, or z bounds");
  Region region;
  for (auto it = node.begin(); it != node.end(); ++it) {
    const auto d = std::string("xyz").find(it.key());
    if (it.key().size() != 1 || d == std::string::npos)
      throw std::runtime_error("unknown trajectory region axis: " + it.key());
    if (!it->is_array() || it->size() != 2 ||
        !(*it)[0].is_number() || !(*it)[1].is_number())
      throw std::runtime_error("trajectory region bounds must be [lo, hi]");
    region.lo[d] = (*it)[0].get<Real>();
    region.hi[d] = (*it)[1].get<Real>();
    if (!std::isfinite(region.lo[d]) || !std::isfinite(region.hi[d]) ||
        region.lo[d] >= region.hi[d])
      throw std::runtime_error("trajectory region requires finite lo < hi");
    region.enabled[d] = true;
  }
  return region;
}

inline Options ParseOptions(const nlohmann::ordered_json& node) {
  Options out;
  if (node.contains("fields")) {
    if (!node["fields"].is_array() || node["fields"].empty())
      throw std::runtime_error("trajectory fields must be a nonempty array");
    out.fields = node["fields"].get<std::vector<std::string>>();
    std::vector<std::string> seen;
    for (const auto& field : out.fields) {
      if (std::find(AllFields().begin(), AllFields().end(), field) == AllFields().end() ||
          std::find(seen.begin(), seen.end(), field) != seen.end())
        throw std::runtime_error("unknown or duplicate trajectory field: " + field);
      seen.push_back(field);
    }
  }
  if (node.contains("max_atoms")) {
    const auto& limit = node["max_atoms"];
    if (!limit.is_number_integer() || limit < 1 || limit > std::numeric_limits<int>::max())
      throw std::runtime_error("trajectory max_atoms must be in [1, INT_MAX]");
    out.max_atoms = limit.get<size_t>();
  }
  if (!node.contains("selection")) return out;
  const auto& selection = node["selection"];
  if (!selection.is_object()) throw std::runtime_error("trajectory selection must be an object");
  for (auto it = selection.begin(); it != selection.end(); ++it)
    if (it.key() != "types" && it.key() != "ids" && it.key() != "region" &&
        it.key() != "regions" && it.key() != "mode")
      throw std::runtime_error("unknown trajectory selection key: " + it.key());
  const auto mode = selection.value("mode", std::string("dynamic"));
  if (mode != "dynamic" && mode != "initial")
    throw std::runtime_error("trajectory selection mode must be dynamic or initial");
  out.initial = mode == "initial";
  if (selection.contains("types")) out.types = ReadIds(selection["types"]);
  if (selection.contains("ids")) {
    out.ids = ReadIds(selection["ids"]);
    out.filter_ids = true;
  }
  if (selection.contains("region") && selection.contains("regions"))
    throw std::runtime_error("use trajectory region or regions, not both");
  if (selection.contains("region")) out.regions.push_back(ReadRegion(selection["region"]));
  if (selection.contains("regions")) {
    const auto& regions = selection["regions"];
    if (!regions.is_array() || regions.empty())
      throw std::runtime_error("trajectory regions must be a nonempty array");
    for (const auto& region : regions) out.regions.push_back(ReadRegion(region));
  }
  if (out.types.empty() && out.ids.empty() && out.regions.empty())
    throw std::runtime_error("trajectory selection must specify types, ids, or region(s)");
  return out;
}
}  // namespace rbmd::trajectory
#undef RBMD_TRAJECTORY_HD
