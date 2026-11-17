# RBMD

Current version: RBMD 2.0

Molecular dynamics software with multi-device support, fully GPU-resident parallelism, and Random Batch algorithms: RBE, RBL, and RBSOG.

RBMD is a GPU-accelerated molecular dynamics package for heterogeneous and high-performance computing systems. It supports NVIDIA CUDA and AMD ROCm/DCU devices, MPI-based multi-GPU execution, and optional NCCL/RCCL collective communication.

## Preprints

For the manuscript of RBMD 2.0, see, [RBMD 2.0: Random batch molecular dynamics package for large-scale simulations on multi-GPU architectures](https://arxiv.org/abs/2609.02694)

## Highlights

- Random Batch Method algorithms for short- and long-range interactions, including RBL, RBE, and RBSOG.
- CUDA and ROCm/DCU backends with explicit wave32 and wave64 configurations.
- Multi-device and multi-node parallelism with simulation data kept on GPUs throughout the main computation path.
- Molecular force fields and interaction models for water, polymers, ionic systems, metals, and three-body materials.
- JSON-based simulation configuration and optional RDF, MSD, and VACF post-processing.

## Build and run

Building RBMD does not require a supercomputer. A Linux workstation or server with a compatible CUDA or ROCm toolchain can build the project and run configurations supported by its available GPU resources. MPI is optional for a single-device build and required for the multi-GPU execution path.

The following example configures a single-GPU CUDA build. Replace the CUDA architecture with the value for your device.

```bash
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

For CUDA, ROCm/DCU, MPI, NCCL/RCCL, compiler selection, CMake options, and troubleshooting, see the [Build and Run Guide](docs/build.en.md) or the [中文构建与运行指南](docs/build.zh_CN.md).

Simulation inputs are JSON files. See the [JSON Configuration Reference](docs/json-configuration.en.md) or [JSON 配置参考](docs/json-configuration.zh_CN.md) for supported fields and examples.

## Benchmarks

The `benchmark/` directory contains curated configurations, Slurm submission scripts, shared input data, RDF results, and result workbooks for three systems:

| System | Records | Workbook |
| --- | --- | --- |
| H2O | [`benchmark/h2o`](benchmark/h2o) | `benchmark/h2o/workbook/run_H2O.xlsx` |
| LJ-salt | [`benchmark/lj_salt`](benchmark/lj_salt) | `benchmark/lj_salt/workbook/run_LJ_salt.xlsx` |
| PEO-LiTFSI | [`benchmark/peo_litfsi`](benchmark/peo_litfsi) | `benchmark/peo_litfsi/workbook/run_PEO.xlsx` |

### Benchmark input data

The benchmark inputs are distributed in two releases.

1. [Original input archive](https://github.com/qizhou1729/rbmd/releases/tag/benchmark-data-v2). Extract `rbmd-data-v2.tar.gz` from the repository root to install the shared water, LJ electrolyte and PEO–LiTFSI inputs and the equilibrated water RDF seed. The 976,200-atom PEO–LiTFSI parent configuration is reused for the revision.
2. [Revised force-benchmark inputs](https://github.com/SOG-AITech/rbmd/releases/tag/benchmark-lj-200000). This release collects the additional full-box configurations for SPC/E water (72,981, 1,970,487 and 15,763,896 atoms) and LJ electrolyte (200,000, 2,000,000 and 28,000,000 atoms). Each configuration is distributed as a separate gzip file. The filename gives the system and atom count.

The same release includes the [revised numerical results and run records](https://github.com/SOG-AITech/rbmd/releases/download/benchmark-lj-200000/rbmd-revision-results.tar.gz). The archive README indexes force errors, physical observables, timing breakdowns and scaling data. For PEO–LiTFSI, see `Zhengzhou/ReviewRBMD/PEO/accuracy/` for force errors, `Zhengzhou/ReviewRBMD/PEO/timing/` for component timings, and the `Zhengzhou/peo_scaling_*` and `Zhengzhou/peo_weak_*` directories for scaling runs.

A supercomputer is not needed to read the benchmark configurations or build RBMD. Reproducing the published multi-device scaling runs does require an HPC environment with the corresponding number of GPUs/DCUs, MPI, and a Slurm-compatible scheduler. Partition names, module versions, device requests, and launcher options in the archived submission scripts are site-specific and must be adapted to the target cluster.

## Repository layout

- `src/`: C++ and GPU implementation.
- `cmake/`: CMake modules and backend configuration.
- `tools/`: bundled third-party source archives used during configuration.
- `examples/`: runnable JSON configurations and input data.
- `benchmark/`: curated H2O, LJ-salt, and PEO-LiTFSI benchmark records.
- `docs/`: English and Chinese build and JSON configuration documentation.
- `scripts/`: analysis, diagnostics, and environment helper utilities.
- `LICENSES/`: third-party license notices.

## License

See [`LICENSE`](LICENSE) for the RBMD project license and [`LICENSES/`](LICENSES) for third-party notices.
