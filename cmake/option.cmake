option(ENABLE_DOUBLE "Enable double, default float" OFF)
option(ENABLE_64BIT_IDS "Enable long long, default int" OFF)
option(USE_ROCM "Use rocm: DCU or AMD" OFF)
option(USE_CUDA "Use cuda: NVIDIA" OFF)
option(USE_MPI "Enable MPI support" OFF)
option(CCL "Enable NCCL/RCCL collective communication support" OFF)

if(NOT DEFINED WARP_SIZE_64)
  message(FATAL_ERROR
          "WARP_SIZE_64 must be explicitly set to ON or OFF. "
          "Example: -DWARP_SIZE_64=ON")
endif()

string(TOUPPER "${WARP_SIZE_64}" WARP_SIZE_64)
if(NOT WARP_SIZE_64 STREQUAL "ON" AND NOT WARP_SIZE_64 STREQUAL "OFF")
  message(FATAL_ERROR
          "WARP_SIZE_64 must be ON or OFF, got: ${WARP_SIZE_64}")
endif()

set(WARP_SIZE_64 "${WARP_SIZE_64}" CACHE STRING
    "Use 64-wide warp/wavefront configuration on TARGET_DCU" FORCE)
set_property(CACHE WARP_SIZE_64 PROPERTY STRINGS ON OFF)
