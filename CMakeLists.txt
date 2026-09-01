cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(rbmd)

#cmake build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type." FORCE)
	message(STATUS "Set CMAKE_BUILD_TYPE \"Release\".")
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cmake)
get_property(cmake_dir GLOBAL PROPERTY CUSTOME_CMAKE_DIR)

#option
include("${cmake_dir}/option.cmake")
include("${cmake_dir}/compile_definitions.cmake")

# 静态编译配置（可选，通过 -DSTATIC_BUILD=ON 启用）
option(STATIC_BUILD "Enable static linking" OFF)
if(STATIC_BUILD)
	include("${cmake_dir}/static_build.cmake")
endif()

if(USE_MPI)
	add_compile_definitions(USE_MPI)
endif()

#hip setting
include("${cmake_dir}/gpu_setting.cmake")

#sub directory
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/common)

file(GLOB_RECURSE output_srcs ./src/output/*.cpp ./src/output/*.cu)
set(src
./src/main.cpp
#./src/executioner/executioner.cpp
./src/Utilities/string_util.cpp
./src/application/application.cpp
./src/application/md_application.cpp
./src/application/postprocess_runner.cpp
./src/parser/command_line.cpp
./src/common/unit_factor.cpp
./src/common/thermo_tui_view.cpp
./src/common/RBEPSample.h
${output_srcs}
		src/common/math_utils.h)

include_directories("${CMAKE_CURRENT_LIST_DIR}/src")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/Utilities")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/parser")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/application")
#include_directories("${CMAKE_CURRENT_LIST_DIR}/src/executioner")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/common")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/output")
if(USE_CUDA)
	set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
	# FTXUI relies on std::variant/std::function patterns that can fail under
	# older NVCC host compilation. Keep the TUI implementation on host C++.
	set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/common/thermo_tui_view.cpp
		PROPERTIES LANGUAGE CXX)
	set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/application/postprocess_runner.cpp
		PROPERTIES LANGUAGE CXX)
endif()


file(GLOB_RECURSE all_include_dirs
    "${CMAKE_CURRENT_LIST_DIR}/src/*.h"
)

set(include_dirs)
foreach(header_file ${all_include_dirs})
    get_filename_component(dir ${header_file} DIRECTORY)
    list(APPEND include_dirs ${dir})
endforeach()
list(REMOVE_DUPLICATES include_dirs)

include_directories(${include_dirs})

#
file(GLOB tools_path ${CMAKE_CURRENT_LIST_DIR}/tools/*.tar.gz)

foreach(tool_path ${tools_path})
	get_filename_component(toolname ${tool_path} NAME)
	get_filename_component(tooldir ${tool_path} PATH)
	string(REGEX REPLACE "\\.tar\\.gz$" "" toolname_without_ext ${toolname})
	set(TOOL_INSTALL_PATH ${tooldir}/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE}/${toolname_without_ext})
	list(APPEND CMAKE_PREFIX_PATH ${TOOL_INSTALL_PATH})
endforeach()

find_package(PkgConfig REQUIRED)
pkg_check_modules(SPDLOG REQUIRED spdlog IMPORTED_TARGET GLOBAL)
pkg_check_modules(CXXOPTS REQUIRED cxxopts IMPORTED_TARGET GLOBAL)
pkg_check_modules(FTXUI REQUIRED ftxui IMPORTED_TARGET GLOBAL)


add_executable(${PROJECT_NAME} ${src})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory
		$<TARGET_FILE_DIR:${PROJECT_NAME}>/analysis
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_LIST_DIR}/scripts/postprocess_freud.py
		$<TARGET_FILE_DIR:${PROJECT_NAME}>/analysis/postprocess_freud.py
	COMMENT "Copying analysis sidecar script")

set(LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/lib)

set(LIBRARY_NAMES
		data_manager
		reader
		neighbor_list
		force
		simulate_pipeline
)
if(USE_CUDA)
	target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARY_NAMES}  PkgConfig::SPDLOG PkgConfig::CXXOPTS PkgConfig::FTXUI)
