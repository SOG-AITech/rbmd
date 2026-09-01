project(reader)

get_property(cmake_dir GLOBAL PROPERTY CUSTOME_CMAKE_DIR)
include(${cmake_dir}/cpp_setting.cmake)

get_property(sppdlog_include GLOBAL PROPERTY SPDLOG_INCLUDE_PATH)
get_property(sppdlog_link GLOBAL PROPERTY SPDLOG_LINK_PATH)
#get_property(data_manager_include GLOBAL PROPERTY DATA_MANAGER_INCLUDE_PATH)
get_property(library_link GLOBAL PROPERTY LIBRARY_DIR)

set(INCLUDE_DIR "../data_manager/include" "../common" )
set(LINK_DIR  ${library_link})
set(DEPENDS_NAME  "data_manager")
if(USE_MPI)
  list(APPEND INCLUDE_DIR "../parallel/include")
  list(APPEND DEPENDS_NAME "parallel")
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()
#message(WARNING  "sppdlog_link: ${sppdlog_link}")
cpp_library(
name ${PROJECT_NAME}
depends_include ${INCLUDE_DIR}
depends_link_dir ${LINK_DIR}
depends_name ${DEPENDS_NAME}
)
set(LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../lib)
target_link_libraries(${PROJECT_NAME}  data_manager   stdc++fs)
if(USE_MPI)
  target_compile_definitions(${PROJECT_NAME} PUBLIC READER_ENABLE_MPI)
  target_link_libraries(${PROJECT_NAME} parallel MPI::MPI_CXX)
endif()
