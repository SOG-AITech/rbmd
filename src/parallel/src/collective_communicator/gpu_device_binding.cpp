#include "gpu_device_binding.h"

#include "common/rbmd_define.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace rbmd {
namespace {

const char* GetEnv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' ? value : nullptr;
}

bool ParseInt(const char* text, int* value) {
  if (text == nullptr || value == nullptr) return false;
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text) return false;
  while (*end != '\0') {
    if (!std::isspace(static_cast<unsigned char>(*end))) return false;
    ++end;
  }
  if (parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool EnvEnabled(const char* name) {
  const char* value = GetEnv(name);
  if (value == nullptr) return false;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "off") != 0 &&
         std::strcmp(value, "OFF") != 0;
}

std::string EnvValueForLog(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : "<unset>";
}

int QueryVisibleDeviceCountOrAbort(const char* caller) {
  int device_count = 0;
#if defined(__CUDA)
  const ERROR_T err = cudaGetDeviceCount(&device_count);
#elif defined(__ROCM)
  const ERROR_T err = hipGetDeviceCount(&device_count);
#else
  const ERROR_T err = SUCCESS;
#endif
  if (err != SUCCESS) {
    std::fprintf(stderr,
                 "[%s] failed to query visible GPU count: %s (%s, code=%d)\n",
                 caller, GETERRORSTRING(err), GETERRORNAME(err),
                 static_cast<int>(err));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
  }
  return device_count;
}

[[noreturn]] void AbortInvalidDeviceBinding(const char* caller,
                                            const GpuDeviceBinding& binding,
                                            const char* reason) {
  std::fprintf(
      stderr,
      "[%s] invalid GPU binding: %s rank=%d/%d local_rank=%d "
      "visible_device_count=%d selected_device=%d device_source=%s "
      "local_rank_source=%s CUDA_VISIBLE_DEVICES=%s HIP_VISIBLE_DEVICES=%s "
      "ROCR_VISIBLE_DEVICES=%s RBMD_GPU_LIST=%s RBMD_DEVICE_LIST=%s "
      "RBMD_GPU_FIRST_ID=%s RBMD_DEVICE_FIRST_ID=%s RBMD_GPU_COUNT=%s "
      "RBMD_DEVICE_COUNT=%s\n",
      caller, reason, binding.current_rank, binding.total_ranks,
      binding.local_rank, binding.visible_device_count, binding.selected_device,
      binding.device_source.c_str(), binding.local_rank_source.c_str(),
      EnvValueForLog("CUDA_VISIBLE_DEVICES").c_str(),
      EnvValueForLog("HIP_VISIBLE_DEVICES").c_str(),
      EnvValueForLog("ROCR_VISIBLE_DEVICES").c_str(),
      EnvValueForLog("RBMD_GPU_LIST").c_str(),
      EnvValueForLog("RBMD_DEVICE_LIST").c_str(),
      EnvValueForLog("RBMD_GPU_FIRST_ID").c_str(),
      EnvValueForLog("RBMD_DEVICE_FIRST_ID").c_str(),
      EnvValueForLog("RBMD_GPU_COUNT").c_str(),
      EnvValueForLog("RBMD_DEVICE_COUNT").c_str());
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  std::abort();
}

int DetectLocalRank(std::string* source) {
  const char* env_names[] = {"SLURM_LOCALID",
                             "FLUX_TASK_LOCAL_ID",
                             "MPT_LRANK",
                             "MV2_COMM_WORLD_LOCAL_RANK",
                             "OMPI_COMM_WORLD_LOCAL_RANK",
                             "PMI_LOCAL_RANK",
                             "PALS_LOCAL_RANKID"};

  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local_comm) == MPI_SUCCESS &&
      local_comm != MPI_COMM_NULL) {
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    if (source != nullptr) {
      *source = "MPI_COMM_TYPE_SHARED";
      for (const char* name : env_names) {
        int env_local_rank = -1;
        if (ParseInt(GetEnv(name), &env_local_rank) &&
            env_local_rank != local_rank) {
          *source += std::string(";ignored_") + name + "=" +
                     std::to_string(env_local_rank);
          break;
        }
      }
    }
    return local_rank;
  }

  for (const char* name : env_names) {
    int local_rank = -1;
    if (ParseInt(GetEnv(name), &local_rank) && local_rank >= 0) {
      if (source != nullptr) *source = name;
      return local_rank;
    }
  }

  if (source != nullptr) *source = "fallback-single-process";
  return 0;
}

std::vector<int> ParseDeviceList(const char* text) {
  std::vector<int> devices;
  if (text == nullptr) return devices;

  std::string normalized(text);
  for (char& c : normalized) {
    if (c == ',' || c == ';' || c == ':') c = ' ';
  }

  std::istringstream input(normalized);
  std::string token;
  while (input >> token) {
    int device = -1;
    if (!ParseInt(token.c_str(), &device) || device < 0) {
      devices.clear();
      return devices;
    }
    devices.push_back(device);
  }
  return devices;
}

bool HasDuplicateDevice(const std::vector<int>& devices) {
  for (size_t i = 0; i < devices.size(); ++i) {
    for (size_t j = i + 1; j < devices.size(); ++j) {
      if (devices[i] == devices[j]) return true;
    }
  }
  return false;
}

void SelectDevice(const char* caller, GpuDeviceBinding* binding) {
  const char* list_text = GetEnv("RBMD_GPU_LIST");
  const char* list_name = "RBMD_GPU_LIST";
  if (list_text == nullptr) {
    list_text = GetEnv("RBMD_DEVICE_LIST");
    list_name = "RBMD_DEVICE_LIST";
  }

  std::vector<int> device_list = ParseDeviceList(list_text);
  if (list_text != nullptr && device_list.empty()) {
    binding->device_source = list_name;
    AbortInvalidDeviceBinding(caller, *binding, "failed to parse GPU list");
  }
  if (!device_list.empty()) {
    if (HasDuplicateDevice(device_list)) {
      binding->device_source = list_name;
      AbortInvalidDeviceBinding(caller, *binding,
                                "GPU list contains duplicate devices");
    }
    for (int device : device_list) {
      if (device < 0 || device >= binding->visible_device_count) {
        binding->selected_device = device;
        binding->device_source = list_name;
        AbortInvalidDeviceBinding(caller, *binding,
                                  "GPU list contains an invisible device");
      }
    }
    binding->explicit_device_selection = true;
    binding->device_pool_size = static_cast<int>(device_list.size());
    binding->selected_device =
        device_list[binding->local_rank % binding->device_pool_size];
    binding->rank_sharing_device_pool =
        binding->local_rank >= binding->device_pool_size;
    binding->device_source = list_name;
    return;
  }

  int first_device = -1;
  const char* first_text = GetEnv("RBMD_GPU_FIRST_ID");
  const char* first_name = "RBMD_GPU_FIRST_ID";
  if (first_text == nullptr) {
    first_text = GetEnv("RBMD_DEVICE_FIRST_ID");
    first_name = "RBMD_DEVICE_FIRST_ID";
  }
  if (first_text != nullptr) {
    if (!ParseInt(first_text, &first_device) || first_device < 0) {
      binding->device_source = first_name;
      AbortInvalidDeviceBinding(caller, *binding,
                                "failed to parse first GPU id");
    }
    int requested_count = binding->visible_device_count - first_device;
    const char* count_text = GetEnv("RBMD_GPU_COUNT");
    const char* count_name = "RBMD_GPU_COUNT";
    if (count_text == nullptr) {
      count_text = GetEnv("RBMD_DEVICE_COUNT");
      count_name = "RBMD_DEVICE_COUNT";
    }
    if (count_text != nullptr &&
        (!ParseInt(count_text, &requested_count) || requested_count <= 0)) {
      binding->device_source = count_name;
      AbortInvalidDeviceBinding(caller, *binding, "failed to parse GPU count");
    }
    if (requested_count <= 0 ||
        first_device + requested_count > binding->visible_device_count) {
      binding->selected_device = first_device;
      binding->device_pool_size = requested_count;
      binding->device_source =
          count_text != nullptr ? std::string(first_name) + "+" + count_name
                                : first_name;
      AbortInvalidDeviceBinding(caller, *binding,
                                "GPU first/count exceeds visible devices");
    }
    binding->explicit_device_selection = true;
    binding->device_pool_size = requested_count;
    binding->selected_device =
        first_device + (binding->local_rank % binding->device_pool_size);
    binding->rank_sharing_device_pool =
        binding->local_rank >= binding->device_pool_size;
    binding->device_source =
        count_text != nullptr ? std::string(first_name) + "+" + count_name
                              : first_name;
    return;
  }

  binding->device_pool_size = binding->visible_device_count;
  binding->selected_device = binding->local_rank % binding->device_pool_size;
  binding->rank_sharing_device_pool =
      binding->local_rank >= binding->device_pool_size;
  binding->device_source = "local_rank%visible_device_count";
}

void ValidateSelectedDeviceOrAbort(const char* caller,
                                   const GpuDeviceBinding& binding) {
  if (binding.visible_device_count <= 0) {
    AbortInvalidDeviceBinding(caller, binding, "no visible GPU device");
  }
  if (binding.device_pool_size <= 0) {
    AbortInvalidDeviceBinding(caller, binding, "empty GPU device pool");
  }
  if (binding.selected_device < 0 ||
      binding.selected_device >= binding.visible_device_count) {
    AbortInvalidDeviceBinding(caller, binding,
                              "selected GPU is outside visible device range");
  }
}

void CheckNodeDeviceSharingOrAbort(const char* caller,
                                   GpuDeviceBinding* binding) {
  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local_comm) != MPI_SUCCESS ||
      local_comm == MPI_COMM_NULL) {
    return;
  }

  int local_size = 1;
  MPI_Comm_size(local_comm, &local_size);
  std::vector<int> selected_devices(local_size, -1);
  MPI_Allgather(&binding->selected_device, 1, MPI_INT,
                selected_devices.data(), 1, MPI_INT, local_comm);
  MPI_Comm_free(&local_comm);

  int same_device_count = 0;
  for (int device : selected_devices) {
    if (device == binding->selected_device) ++same_device_count;
  }
  binding->local_node_size = local_size;
  binding->same_device_rank_count = same_device_count;
  binding->selected_device_shared_on_node = same_device_count > 1;

  if (EnvEnabled("RBMD_GPU_BINDING_STRICT") &&
      binding->selected_device_shared_on_node) {
    AbortInvalidDeviceBinding(
        caller, *binding,
        "strict GPU binding rejects multiple local ranks on one device");
  }
}

void WarnIfDeviceLooksShared(const char* caller,
                             const GpuDeviceBinding& binding) {
  static bool warned_this_process = false;
  if (warned_this_process) return;
  warned_this_process = true;

  if (binding.rank_sharing_device_pool) {
    std::fprintf(stderr,
                 "[%s] warning: rank=%d local_rank=%d maps onto device=%d "
                 "with only %d device(s) in the selected pool; multiple ranks "
                 "on this node may share a GPU.\n",
                 caller, binding.current_rank, binding.local_rank,
                 binding.selected_device, binding.device_pool_size);
  }
  if (binding.selected_device_shared_on_node) {
    std::fprintf(stderr,
                 "[%s] warning: rank=%d selected device=%d is selected "
                 "by %d local rank(s) on this node (local_node_size=%d). Set "
                 "RBMD_GPU_BINDING_STRICT=1 to make this fatal.\n",
                 caller, binding.current_rank, binding.selected_device,
                 binding.same_device_rank_count, binding.local_node_size);
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
#if defined(__CUDA)
  const ERROR_T mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
#elif defined(__ROCM)
  const ERROR_T mem_err = hipMemGetInfo(&free_bytes, &total_bytes);
#else
  const ERROR_T mem_err = SUCCESS;
#endif
  if (mem_err != SUCCESS || total_bytes == 0) return;

  int warn_used_mb = 1024;
  ParseInt(GetEnv("RBMD_GPU_SHARED_WARN_USED_MB"), &warn_used_mb);
  if (warn_used_mb < 0) return;

  const size_t used_bytes = total_bytes - free_bytes;
  const size_t used_mb = used_bytes / (1024 * 1024);
  const size_t total_mb = total_bytes / (1024 * 1024);
  const size_t free_mb = free_bytes / (1024 * 1024);
  if (used_mb >= static_cast<size_t>(warn_used_mb)) {
    std::fprintf(stderr,
                 "[%s] warning: selected device=%d already has about "
                 "%zu MiB used (%zu MiB free / %zu MiB total) at startup; "
                 "this may indicate a shared or busy GPU. Adjust "
                 "RBMD_GPU_LIST, RBMD_GPU_FIRST_ID, or scheduler GPU binding "
                 "if this is unexpected.\n",
                 caller, binding.selected_device, used_mb, free_mb, total_mb);
  }
}

void PrintVerboseBinding(const char* caller, const GpuDeviceBinding& binding) {
  if (!EnvEnabled("RBMD_GPU_BINDING_VERBOSE") &&
      !EnvEnabled("RBMD_DEBUG_GPU_BINDING") &&
      !binding.explicit_device_selection) {
    return;
  }
  std::fprintf(stderr,
               "[%s] GPU binding: rank=%d/%d local_rank=%d "
               "local_rank_source=%s visible_device_count=%d "
               "selected_device=%d device_source=%s local_node_size=%d "
               "same_device_rank_count=%d CUDA_VISIBLE_DEVICES=%s "
               "HIP_VISIBLE_DEVICES=%s ROCR_VISIBLE_DEVICES=%s\n",
               caller, binding.current_rank, binding.total_ranks,
               binding.local_rank, binding.local_rank_source.c_str(),
               binding.visible_device_count, binding.selected_device,
               binding.device_source.c_str(), binding.local_node_size,
               binding.same_device_rank_count,
               EnvValueForLog("CUDA_VISIBLE_DEVICES").c_str(),
               EnvValueForLog("HIP_VISIBLE_DEVICES").c_str(),
               EnvValueForLog("ROCR_VISIBLE_DEVICES").c_str());
}

}  // namespace

GpuDeviceBinding BindGpuDeviceForRankOrAbort(const char* caller,
                                             int current_rank,
                                             int total_ranks) {
  GpuDeviceBinding binding;
  binding.current_rank = current_rank;
  binding.total_ranks = total_ranks;
  binding.local_rank = DetectLocalRank(&binding.local_rank_source);
  binding.visible_device_count = QueryVisibleDeviceCountOrAbort(caller);
  if (binding.visible_device_count <= 0) {
    AbortInvalidDeviceBinding(caller, binding, "no visible GPU device");
  }

  SelectDevice(caller, &binding);
  ValidateSelectedDeviceOrAbort(caller, binding);
  CheckNodeDeviceSharingOrAbort(caller, &binding);

  const ERROR_T err = SET_DEVICE(binding.selected_device);
  if (err != SUCCESS) {
    std::fprintf(stderr,
                 "[%s] failed to bind rank=%d/%d local_rank=%d to device=%d: "
                 "%s (%s, code=%d)\n",
                 caller, current_rank, total_ranks, binding.local_rank,
                 binding.selected_device, GETERRORSTRING(err),
                 GETERRORNAME(err), static_cast<int>(err));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
  }

  PrintVerboseBinding(caller, binding);
  WarnIfDeviceLooksShared(caller, binding);
  return binding;
}

}  // namespace rbmd
