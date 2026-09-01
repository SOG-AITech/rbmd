# RBMD Build and Run Guide

[中文](build.zh_CN.md) · [JSON Configuration Reference](json-configuration.en.md)

This guide explains how to build and run RBMD with CUDA or ROCm/DCU. The current source tree produces an executable named `rbmd`.

## Requirements

| Component | Requirement | Notes |
| --- | --- | --- |
| Operating system | Linux recommended | The build and examples target Linux/HPC environments. |
| Build tools | CMake ≥ 3.16, a C++17 compiler, and `pkg-config` | The minimum CMake version is declared by the project. |
| CUDA build | CUDA Toolkit | Enable with `USE_CUDA=ON`. |
| ROCm build | ROCm/HIP, rocthrust, and hipcub | Enable with `USE_ROCM=ON`. |
| Multi-GPU parallelism | MPI (OpenMPI recommended) | Enable with `USE_MPI=ON`. |
| GPU collective communication | NCCL for CUDA or RCCL for ROCm | Required only with `CCL=ON`. |

On Ubuntu/Debian, install the base build dependencies as needed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config

# Required only for multi-GPU/MPI builds
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

Install the CUDA Toolkit, ROCm, and compatible GPU drivers through your distribution or hardware vendor. During the first CMake configure, the bundled archives in `tools/` are used to build spdlog, cxxopts, and FTXUI; no separate download is required.

## Quick Start

The following Linux example targets an NVIDIA GPU. Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability of your GPU: `80` for A100, `86` for RTX 30 series, and `89` for RTX 40 series.

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

## Build

Select exactly one GPU backend per build: CUDA or ROCm. `-DWARP_SIZE_64=ON` or `OFF` must always be passed explicitly; it is a required project option.

### CUDA

Single-GPU (or no MPI) build:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_MPI=OFF \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --parallel
```

Multi-GPU/MPI build:

```bash
cmake -S . -B build-cuda-mpi \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_MPI=ON \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-mpi --parallel
```

When CUDA or MPI is outside CMake's default search paths, provide the compiler locations explicitly:

```bash
cmake -S . -B build-cuda-mpi \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_C_COMPILER="$(command -v mpicc)" \
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DUSE_CUDA=ON -DUSE_MPI=ON \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

To compile the project's collective-communication path, enable `CCL` and point CMake to the NCCL installation:

```bash
cmake -S . -B build-cuda-ccl \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_MPI=ON -DCCL=ON \
  -DWARP_SIZE_64=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNCCL_ROOT=/opt/nccl
cmake --build build-cuda-ccl --parallel
```

Setting only `NCCL_ROOT` does not enable the collective-communication path; `CCL=ON` is also required. For non-standard NCCL layouts, set `NCCL_INCLUDE_DIR` and `NCCL_LIB_DIR` to the header and library directories.

### ROCm / DCU

Ensure `hipcc` is on `PATH`, then set the GFX architecture for the target GPU. Use `rocminfo | rg 'Name:'` to inspect the architecture identifier.

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

`WARP_SIZE_64=ON` is appropriate for AMD GPUs/DCUs with a 64-wide wavefront; CUDA builds usually use `OFF`. If HIP is outside the default path, also pass `-DHIP_PATH=/opt/rocm/hip`. With `-DCCL=ON`, a ROCm build searches for RCCL.

### Common CMake Options

| Option | Default | Description |
| --- | --- | --- |
| `USE_CUDA` | `OFF` | Enables the NVIDIA CUDA backend. |
| `USE_ROCM` | `OFF` | Enables the AMD ROCm/DCU backend. |
| `USE_MPI` | `OFF` | Enables MPI parallelism. |
| `CCL` | `OFF` | Enables NCCL/RCCL collective-communication definitions and dependencies. |
| `WARP_SIZE_64` | No default; required | `ON` selects the wave64 configuration; `OFF` is typical for CUDA. |
| `ENABLE_DOUBLE` | `OFF` | Uses double precision; single precision is the default. |
| `ENABLE_64BIT_IDS` | `OFF` | Uses 64-bit atom/index IDs. |
| `CMAKE_BUILD_TYPE` | `Release` | Can be set to `Debug`, `Release`, and other CMake build types. |

## Run

Start RBMD with a JSON configuration file:

```bash
./build/rbmd -j path/to/rbmd.json
```

| Option | Description |
| --- | --- |
| `-j, --json FILE` | Required. Path to the JSON configuration file. |
| `-t, --tui` | Enables the FTXUI terminal dashboard. |
| `-h, --help` | Prints help. |
| `-v, --version` | Prints version information. |

Paths relative to a configuration are resolved from the configuration file's directory. You can therefore run an example directly from the repository root:

```bash
# Run the EAM copper example from the repository root
./build/rbmd -j examples/03-metal/Cu/Cu.json

# Run with two MPI ranks; adjust CUDA_VISIBLE_DEVICES and -np to the hardware
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 \
  ./build-cuda-mpi/rbmd -j examples/01-peo/rbmd.replicate-2x1x1.json
```

MPI runs retain standard output only on the root rank by default, preventing interleaved terminal logs. For troubleshooting, set `RBMD_DEBUG_KEEP_ALL_RANK_STDIO=1` to retain output from every rank.

### Outputs and Post-processing

- Terminal output and `rbmd.log`: thermodynamic information and runtime logs.
- `rbmd.trj`: trajectory file, created when `outputs.trajectory_out` is enabled.
- `analysis/`: `postprocess_freud.py` is placed alongside the executable at build time. When RDF, MSD, or VACF is configured, the program can invoke this script after the simulation finishes.

For field semantics, required keys, and output control, see the [JSON Configuration Reference](json-configuration.en.md).

## Examples

| Directory | System / features |
| --- | --- |
| [01-peo](../examples/01-peo) | PEO + Li, CVFF and RBE; includes a 2×1×1 replication configuration. |
| [02-water-methane](../examples/02-water-methane) | Water–methane system, CVFF and RBE. |
| [03-metal](../examples/03-metal) | Cu, Ag, and Au metal systems using EAM potentials. |
| [04-threebody](../examples/04-threebody) | Si and hBN three-body systems using Tersoff potentials. |
| [ljsalt](../examples/ljsalt) | Lennard-Jones salt-water system using `LJ/CUT/COUL/LONG` and RBE. |

## Troubleshooting

### CMake reports `WARP_SIZE_64 must be explicitly set`

Add `-DWARP_SIZE_64=OFF` (normally CUDA) or `-DWARP_SIZE_64=ON` (wave64 AMD/DCU) to the configure command.

### CUDA reports an incompatible GPU architecture

Reconfigure with `CMAKE_CUDA_ARCHITECTURES` set to the target GPU's compute capability. Do not reuse an architecture value from another machine: RTX 3080/3090 uses `86`, whereas A100 uses `80`.

### MPI, NCCL, or RCCL cannot be found

Ensure that the matching development package and compiler wrappers are loaded. For MPI builds, pass `mpicc` and `mpicxx` explicitly if necessary. CUDA collective communication can be located through `NCCL_ROOT` or `NCCL_INCLUDE_DIR` / `NCCL_LIB_DIR`. When the respective features are not needed, keep `USE_MPI=OFF` and `CCL=OFF`.

### The configuration is found, but data or potential files are not

Check paths relative to the JSON file—not the shell working directory. Keeping the data files, potential files, and JSON file in the same example directory is recommended.
