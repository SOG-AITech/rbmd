# Release Builds

触发方式：
- 推送 `v*` 标签（如 `v1.0.0`）
- 手动触发 (`workflow_dispatch`)

## 构建目标

| Job | 系统 | GPU 架构 | 输出包名 |
|-----|------|---------|---------|
| cuda-10 | Ubuntu 18.04 | sm_30~75 (Kepler ~ Turing) | `rbmd-gpu-*-cuda10.2-ubuntu18.04-x64.tar.gz` |
| cuda-11 | Ubuntu 20.04 | sm_60~89 (Pascal ~ Ada) | `rbmd-gpu-*-cuda11.8-ubuntu20.04-x64.tar.gz` |
| cuda-12 | Ubuntu 20.04 | sm_89~120 (Ada ~ Blackwell) | `rbmd-gpu-*-cuda12.8-ubuntu20.04-x64.tar.gz` |
| rocm | Ubuntu 20.04 | gfx1030~1102 (RX 6000/7000) | `rbmd-gpu-*-rocm6.2-ubuntu20.04-x64.tar.gz` |
| rocm-warp64 | Ubuntu 20.04 | gfx1030~1102 (DCU/wave64) | `rbmd-gpu-*-rocm6.2-warp64-ubuntu20.04-x64.tar.gz` |

## GPU 架构对照

### NVIDIA CUDA

| 架构代号 | sm_ | GPU 型号 |
|---------|-----|---------|
| Kepler | 30, 35, 37 | GTX 600/700 系列 |
| Maxwell | 50, 52, 53 | GTX 750, 900 系列 |
| Pascal | 60, 61, 62 | GTX 10 系列 |
| Volta | 70, 72 | Titan V, V100 |
| Turing | 75 | RTX 20 系列, GTX 16 系列 |
| Ampere | 80, 86 | RTX 30 系列, A100 |
| Ada Lovelace | 89 | RTX 40 系列 |
| Blackwell | 90, 120 | RTX 50 系列 |

### AMD ROCm

| gfx_ | GPU 型号 |
|------|---------|
| gfx1030, gfx1031, gfx1032 | RX 6600/6700/6800/6900 系列 |
| gfx1100, gfx1101, gfx1102 | RX 7600/7700/7800/7900 系列 |

**WARP_SIZE_64**：AMD GPU 默认 wavefront 宽度为 64，部分 AMD GPU 和 DCU 需要此选项。

## 依赖

| 版本 | MPI | 集合通信库 |
|------|-----|----------|
| CUDA | OpenMPI + CUDA 支持 | NCCL |
| ROCm | OpenMPI + ROCm 支持 | RCCL |

## 包内容

```
rbmd-gpu-<version>-<backend>-<os>-x64.tar.gz
├── bin/
│   └── rbmd-gpu          # 可执行文件
├── examples/             # 示例目录
├── README.md
└── LICENSE
```

## 下载建议

| 你的 GPU | 推荐下载 |
|---------|---------|
| GTX 600~900 | cuda10.2 |
| GTX 10 系列 | cuda11.8 |
| RTX 20/30 系列 | cuda11.8 |
| RTX 40 系列 | cuda11.8 或 cuda12.8 |
| RTX 50 系列 | cuda12.8 |
| AMD RX 6000/7000 | rocm6.2 |
| DCU 或 wave64 AMD | rocm6.2-warp64 |
