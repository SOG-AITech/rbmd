#pragma once
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <string>
#include "data_manager.h"

namespace rbmd { namespace revision {
// Read once, before any force/controller initialization. All ranks use the same JSON.
struct Options {
  bool enabled = false, isolated_timing = false, forces = false, energy = false;
  bool thermostat_heat = false, unmodified_nve = false;
  int interval = 1, fixed_samples = 0;
  std::string prefix = "revision";
  std::string reference_force_file;
  std::int64_t reference_timestep = -1, reference_id_offset = 0;
  double reference_position_tolerance = 1.e-6;
};
inline const Options& GetOptions() {
  static const Options options = [] {
    Options o;
    const auto config = DataManager::getInstance().getConfigData();
    if (!config->PathExists({"execution", "diagnostics"})) return o;
    const auto& j = config->GetJsonNode("execution").at("diagnostics");
    if (!j.is_object()) throw std::runtime_error("execution.diagnostics must be an object");
    for (auto it = j.begin(); it != j.end(); ++it) {
      const auto& k = it.key();
      if (k != "isolated_timing" && k != "forces" && k != "energy" &&
          k != "thermostat_heat" && k != "unmodified_nve" &&
          k != "interval" && k != "fixed_samples" && k != "prefix" &&
          k != "reference_force_file" && k != "reference_timestep" &&
          k != "reference_id_offset" && k != "reference_position_tolerance")
        throw std::runtime_error("Unknown diagnostics option: " + k);
    }
    o.enabled = true;
    o.isolated_timing = j.value("isolated_timing", false);
    o.forces = j.value("forces", false);
    o.energy = j.value("energy", false);
    o.thermostat_heat = j.value("thermostat_heat", false);
    o.unmodified_nve = j.value("unmodified_nve", false);
    for (const char* key : {"interval", "fixed_samples"}) {
      if (j.contains(key) && (!j.at(key).is_number_integer() ||
          j.at(key).get<std::int64_t>() < 0 ||
          j.at(key).get<std::int64_t>() > std::numeric_limits<int>::max()))
        throw std::runtime_error(std::string("diagnostics requires a nonnegative int: ") + key);
    }
    o.interval = j.value("interval", 1);
    o.fixed_samples = j.value("fixed_samples", 0);
    o.prefix = j.value("prefix", std::string("revision"));
    o.reference_force_file = j.value("reference_force_file", std::string());
    for (const char* key : {"reference_timestep", "reference_id_offset"})
      if (j.contains(key) && (!j.at(key).is_number_integer() ||
          (j.at(key).is_number_unsigned() && j.at(key).get<std::uint64_t>() >
           static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))))
        throw std::runtime_error(std::string("diagnostics requires an integer: ") + key);
    o.reference_timestep = j.value("reference_timestep", std::int64_t(-1));
    o.reference_id_offset = j.value("reference_id_offset", std::int64_t(0));
    o.reference_position_tolerance = j.value("reference_position_tolerance", 1.e-6);
    if (o.reference_timestep < -1 || !(o.reference_position_tolerance >= 0) ||
        o.reference_position_tolerance > std::numeric_limits<double>::max())
      throw std::runtime_error("Invalid reference timestep or position tolerance");
    if (!o.reference_force_file.empty() && !o.fixed_samples)
      throw std::runtime_error("A fixed reference force requires fixed_samples > 0");
    if (o.interval <= 0 || o.fixed_samples < 0 || o.prefix.empty())
      throw std::runtime_error("diagnostics: interval > 0, fixed_samples >= 0, nonempty prefix required");
    if (o.fixed_samples && ((!o.forces && o.reference_force_file.empty()) || o.interval != 1 || o.thermostat_heat))
      throw std::runtime_error("fixed_samples requires forces or reference_force_file, interval=1, thermostat_heat=false");
    return o;
  }();
  return options;
}
}} // namespace rbmd::revision
