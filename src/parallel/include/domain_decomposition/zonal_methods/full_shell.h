#pragma once

#include "halo_region.h"
#include "zonal_method.h"

class FullShell : public ZonalMethod {
 public:
  FullShell() = default;
  ~FullShell() override = default;

  //! 这里是否与设置的那个那个cell count有关系？ 26 好像没有关系？因为有width
  /**
   * Returns up to 26 halo Regions of the process.
   * If a process is spanning a whole dimension, then fewer regions can be
   * returned. The regions indicate, where the processes lie that require halo
   * copies from the current process.
   * @param initialRegion boundary of the current process
   * @param cutoffRadius
   * @return vector of regions
   */
  std::vector<HaloRegion> getHaloImportForceExportRegions(
      HaloRegion& initialRegion, double cutoffRadius, bool coversWholeDomain[3],
      double cellLength[3]) override {
    auto condition = [](const int[3]) -> bool {
      // no condition for leaving particles.
      return true;
    };
    return getHaloRegionsConditional(initialRegion, cutoffRadius,
                                     coversWholeDomain, condition);
  }

  std::vector<HaloRegion> getHaloExportForceImportRegions(
      HaloRegion& initialRegion, double cutoffRadius, bool coversWholeDomain[3],
      double cellLength[3]) override {
    auto condition = [](const int[3]) -> bool {
      // no condition for leaving particles.
      return true;
    };
    return getHaloRegionsConditionalInside(initialRegion, cutoffRadius,
                                           coversWholeDomain, condition);
  }
};
