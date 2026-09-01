# RBMD 构建与运行指南

[English](build.en.md) · [JSON 配置参考](json-configuration.zh_CN.md)

本文档说明如何在 CUDA 或 ROCm/DCU 环境中构建和运行 RBMD。源代码当前生成的可执行文件名为 `rbmd`。

## 依赖

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| 操作系统 | 推荐 Linux | 当前构建与示例主要面向 Linux/HPC 环境。 |
| 构建工具 | CMake ≥ 3.16、支持 C++17 的编译器、`pkg-config` | CMake 最低版本由项目声明。 |
| CUDA 构建 | CUDA Toolkit | 使用 `USE_CUDA=ON`。 |
| ROCm 构建 | ROCm/HIP、rocthrust、hipcub | 使用 `USE_ROCM=ON`。 |
| 多 GPU 并行 | MPI（推荐 OpenMPI） | 使用 `USE_MPI=ON`。 |
| GPU 集合通信 | NCCL（CUDA）或 RCCL（ROCm） | 仅在 `CCL=ON` 时需要。 |

Ubuntu/Debian 上的基础构建依赖可按需安装：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config

# 仅多 GPU/MPI 构建需要
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

CUDA Toolkit、ROCm 和对应 GPU 驱动应通过发行版或硬件厂商提供的方式安装，并保证编译器与运行时版本兼容。第一次配置时，CMake 会从仓库的 `tools/` 压缩包中构建 spdlog、cxxopts 和 FTXUI，无需单独下载这些依赖。

## 快速开始

以下示例在 Linux 上以 NVIDIA GPU 为例。将 `CMAKE_CUDA_ARCHITECTURES` 改为目标 GPU 的计算能力，例如 A100 为 `80`、RTX 30 系列为 `86`、RTX 40 系列为 `89`。

```bash
git clone https://github.com/GuoYongFa/rbmd-gpu-parallel.git
cd rbmd-gpu-parallel

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_MPI=OFF \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel

./build/rbmd --version
./build/rbmd -j examples/03-metal/Cu/Cu.json
```

## 构建

一次构建只应选择一个 GPU 后端：CUDA 或 ROCm。无论使用哪种后端，都必须明确传入 `-DWARP_SIZE_64=ON` 或 `OFF`；这是项目的强制配置项。

### CUDA

单 GPU（或不启用 MPI）示例：

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_MPI=OFF \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --parallel
```

多 GPU / MPI 示例：

```bash
cmake -S . -B build-cuda-mpi \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_MPI=ON \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-mpi --parallel
```

若 CUDA、MPI 不在默认搜索路径，可补充其编译器路径；例如：

```bash
cmake -S . -B build-cuda-mpi \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_C_COMPILER="$(command -v mpicc)" \
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DUSE_CUDA=ON -DUSE_MPI=ON \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

如需启用项目的集合通信编译路径，还需打开 `CCL`，并向 CMake 提供 NCCL 安装位置：

```bash
cmake -S . -B build-cuda-ccl \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_MPI=ON -DCCL=ON \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNCCL_ROOT=/opt/nccl
cmake --build build-cuda-ccl --parallel
```

如果仅指定 `NCCL_ROOT` 而未设置 `CCL=ON`，项目不会启用该集合通信路径。对于非标准 NCCL 布局，可改用环境变量 `NCCL_INCLUDE_DIR` 与 `NCCL_LIB_DIR` 指定头文件和库目录。

### ROCm / DCU

确认 `hipcc` 位于 `PATH` 后，按目标 GPU 指定 GFX 架构。可用 `rocminfo | rg 'Name:'` 查看设备架构标识。

```bash
cmake -S . -B build-rocm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v hipcc)" \
  -DUSE_ROCM=ON \
  -DUSE_MPI=ON \
  -DWARP_SIZE_64=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906
cmake --build build-rocm --parallel
```

`WARP_SIZE_64=ON` 适用于 wavefront 宽度为 64 的 AMD GPU/DCU；CUDA 构建通常使用 `OFF`。若 HIP 不在默认路径，可额外传入 `-DHIP_PATH=/opt/rocm/hip`。启用 `-DCCL=ON` 时，ROCm 构建会查找 RCCL。

### 常用 CMake 选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `USE_CUDA` | `OFF` | 启用 NVIDIA CUDA 后端。 |
| `USE_ROCM` | `OFF` | 启用 AMD ROCm/DCU 后端。 |
| `USE_MPI` | `OFF` | 启用 MPI 并行。 |
| `CCL` | `OFF` | 启用 NCCL/RCCL 集合通信相关编译定义与依赖。 |
| `WARP_SIZE_64` | 无默认值，必须设置 | `ON` 为 wave64 配置；`OFF` 为 CUDA 常用配置。 |
| `ENABLE_DOUBLE` | `OFF` | 使用双精度；默认单精度。 |
| `ENABLE_64BIT_IDS` | `OFF` | 使用 64 位原子/索引 ID。 |
| `CMAKE_BUILD_TYPE` | `Release` | 可设为 `Debug`、`Release` 等。 |

## 运行

程序通过 JSON 配置文件启动：

```bash
./build/rbmd -j path/to/rbmd.json
```

| 选项 | 说明 |
| --- | --- |
| `-j, --json FILE` | 必需。指定 JSON 配置文件。 |
| `-t, --tui` | 启用 FTXUI 终端仪表盘。 |
| `-h, --help` | 显示帮助。 |
| `-v, --version` | 显示版本。 |

配置中的相对路径按 JSON 文件所在目录解析。因此可以在仓库根目录直接指定示例配置：

```bash
# 在仓库根目录运行 EAM 铜示例
./build/rbmd -j examples/03-metal/Cu/Cu.json

# 以两个 MPI rank 运行；按实际设备数量设置 CUDA_VISIBLE_DEVICES 和 -np
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 \
  ./build-cuda-mpi/rbmd -j examples/01-peo/rbmd.replicate-2x1x1.json
```

默认情况下，MPI 运行仅保留 root rank 的标准输出，避免多 rank 终端日志交错。排查问题时可设置 `RBMD_DEBUG_KEEP_ALL_RANK_STDIO=1` 查看全部 rank 的输出。

### 输出与后处理

- 终端日志和 `rbmd.log`：热力学信息与运行日志。
- `rbmd.trj`：当 `outputs.trajectory_out` 启用时生成的轨迹文件。
- `analysis/`：构建后与可执行文件一同放置 `postprocess_freud.py`；配置了 RDF、MSD 或 VACF 时，程序可在结束阶段调用该后处理脚本。

完整配置语义、必填字段和输出控制见 [JSON 配置参考](json-configuration.zh_CN.md)。

## 示例

| 目录 | 体系 / 特性 |
| --- | --- |
| [01-peo](../examples/01-peo) | PEO + Li，CVFF、RBE；包含 2×1×1 扩胞配置。 |
| [02-water-methane](../examples/02-water-methane) | 水–甲烷体系，CVFF、RBE。 |
| [03-metal](../examples/03-metal) | Cu、Ag、Au 金属体系，EAM 势。 |
| [04-threebody](../examples/04-threebody) | Si 与 hBN 三体体系，Tersoff 势。 |
| [ljsalt](../examples/ljsalt) | Lennard-Jones 盐水体系，`LJ/CUT/COUL/LONG` 与 RBE。 |

## 常见问题

### CMake 报错 `WARP_SIZE_64 must be explicitly set`

在配置命令中补充 `-DWARP_SIZE_64=OFF`（CUDA 通常如此）或 `-DWARP_SIZE_64=ON`（wave64 AMD/DCU）。

### CUDA 编译或运行提示架构不匹配

重新配置并将 `CMAKE_CUDA_ARCHITECTURES` 设置为目标 GPU 的计算能力。不要直接沿用其他机器的数值；例如 RTX 3080/3090 应使用 `86`，而非 A100 的 `80`。

### 找不到 MPI、NCCL 或 RCCL

确认对应开发包和编译器包装器已加载。MPI 构建可显式指定 `mpicc`、`mpicxx`；CUDA 集合通信还可通过 `NCCL_ROOT` 或 `NCCL_INCLUDE_DIR` / `NCCL_LIB_DIR` 提供搜索路径。未使用 MPI 或集合通信时，分别保持 `USE_MPI=OFF`、`CCL=OFF`。

### 配置文件能找到，但 data 或势文件找不到

检查 JSON 内的相对路径是否相对于 JSON 文件本身，而不是当前 shell 的工作目录；建议将 data 文件和势文件与 JSON 放在同一示例目录中。
