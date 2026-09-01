# 静态编译配置 - 适用于 Linux 和 Windows
# 用于生成小体积的静态链接可执行文件

message(STATUS "Configuring for static build")

# 强制静态链接所有库
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

# Linux 静态链接配置
if(UNIX AND NOT APPLE)
    message(STATUS "Configuring Linux static build")

    # 静态链接标准库和系统库
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")

    # MPI/GPU 运行时由用户系统提供，避免在发布包里强行私有化这些库。
    if(USE_MPI)
        message(STATUS "MPI uses system runtime libraries during static packaging")
    endif()

    if(USE_MPI OR USE_CUDA OR USE_ROCM)
        message(STATUS "Keeping GPU/MPI runtime dependencies dynamic during static packaging")
    else()
        # 纯 CPU 静态包优先使用静态库
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

        # pkg-config 强制静态链接
        set(ENV{PKG_CONFIG_ALL_STATIC} "1")
    endif()

    # Release 模式优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(USE_CUDA OR USE_ROCM)
            message(STATUS "GPU static build disables LTO to avoid fatbin/LTO linker collisions")
            add_compile_options(-O3 -ffunction-sections -fdata-sections)
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -s")
        else()
            add_compile_options(-O3 -flto -ffunction-sections -fdata-sections)
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto -Wl,--gc-sections -s")
        endif()
    endif()
endif()

# Windows 静态链接配置
if(WIN32)
    message(STATUS "Configuring Windows static build")

    # 静态链接 C/C++ 运行时库
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    # Release 模式优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Ob2 /Oi /Ot /GL /GS-)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF /OPT:ICF")

        # 移除调试信息
        string(REPLACE "/Zi" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/DEBUG" "" CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    endif()
endif()

# CUDA 静态链接配置
if(USE_CUDA)
    message(STATUS "Configuring CUDA static runtime")
    set(CUDA_USE_STATIC_CUDA_RUNTIME ON CACHE BOOL "Use static CUDA runtime" FORCE)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    if(WIN32)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/MT")
    endif()
endif()

# MPI 静态链接配置
if(USE_MPI)
    message(STATUS "MPI build uses system-provided compilers and runtime libraries")
    if(UNIX)
        # MPI 编译器和运行时由调用方/系统环境提供
    elseif(WIN32)
        # MS-MPI 静态库
        set(MPI_CXX_LIBRARIES "msmpi.lib" CACHE STRING "MPI static library" FORCE)
    endif()
endif()
