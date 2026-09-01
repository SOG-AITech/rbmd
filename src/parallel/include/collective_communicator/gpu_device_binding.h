#pragma once

#include <string>

namespace rbmd {

struct GpuDeviceBinding {
  int current_rank = -1;
  int total_ranks = -1;
  int local_rank = -1;
  int visible_device_count = 0;
  int selected_device = -1;
  int device_pool_size = 0;
  int local_node_size = 1;
  int same_device_rank_count = 1;
  bool explicit_device_selection = false;
  bool rank_sharing_device_pool = false;
  bool selected_device_shared_on_node = false;
  std::string local_rank_source;
  std::string device_source;
};

GpuDeviceBinding BindGpuDeviceForRankOrAbort(const char* caller,
                                             int current_rank,
                                             int total_ranks);

}  // namespace rbmd
