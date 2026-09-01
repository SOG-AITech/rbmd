add_compile_definitions(USE_DOUBLE=${ENABLE_DOUBLE})
add_compile_definitions(USE_64BIT_IDS=${ENABLE_64BIT_IDS})

if(CCL)
  add_compile_definitions(USE_CCL=1)
else()
  add_compile_definitions(USE_CCL=0)
endif()

if(WARP_SIZE_64 STREQUAL "ON")
  add_compile_definitions(WARP_SIZE_64=1)
else()
  add_compile_definitions(WARP_SIZE_64=0)
endif()
