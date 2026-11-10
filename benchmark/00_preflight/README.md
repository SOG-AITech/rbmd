# 00 构建、通信和计时边界

正式实验只跑单精度；双精度仅用于RMS参考力，不做双精度性能或动力学实验。

| 用途 | 可执行文件 | 构建要求 |
|---|---|---|
| RBMD正式实验 | `NewBenchmark/software/rbmd-single/rbmd` | ENABLE_DOUBLE=OFF，64位ID |
| LAMMPS正式实验及后处理 | `~/software/rbm26/build-hip-single/lmp` | KOKKOS_PREC=single、FFT_SINGLE=ON、FFT_KOKKOS=HIPFFT、HIP、MOLECULE、RIGID |
| RMS参考力 | `NewBenchmark/software/lammps-reference-double/lmp` | CPU、KOKKOS关闭、FFT_SINGLE=OFF、run 0 |

```bash
source env.sh
sbatch 00_preflight/build_lammps.slurm production
sbatch 00_preflight/build_lammps.slurm reference
sbatch 00_preflight/build_rbmd.slurm
# 构建成功后再验证
sbatch 00_preflight/verify_lammps.slurm
python3 00_preflight/audit.py --output environment-audit.json
```

正式LAMMPS保留CPU的KISS实现，但GPU PPPM使用hipFFT；实际后端以启动日志为准。旧hipFFT二进制仍在 `~/lammps-stable_22Jul2025_rbm/build/lmp`，本套实验不调用它。

参考力case有 `lammps_role=reference`，用 `sbatch 00_preflight/reference.slurm CASE` 提交：CPU进程不分配HIP设备、不启用Kokkos。郑州该分区要求至少申请1张DCU，因此reference.slurm保留1张资源申请，但CPU参考程序不启用或使用它。正式GPU任务用 `common/job.slurm`。预检按case角色核对构建；RMS参考的数值收敛由01按被测误差比例判断。

LAMMPS使用DTK26.04/GCC12.2/OpenMPI5.0.3运行环境；RBMD沿用DTK25.04.4/GCC11.4及原MPI模块。每次run_receipt记录实际MPI、HIP版本、动态库、二进制哈希和环境，论文应披露差异。LAMMPS请求 `gpu/aware on`，水SHAKE不再用 `suffix off`，由 `-sf kk` 选择SHAKE/kk。`verify_lammps.slurm`是两节点设备缓冲区MPI短测试，不替代水约束或大规模容量测试。

RBMD构建先复制src/cmake/tools及必需的scripts/postprocess_freud.py到 `software/source-timing`，原仓库不改。补丁仅使计时额外等待受 `isolated_timing` 控制，并在固定构型验力时读取实际RBL壳层列表长度。保留窗口边界同步和算法所需的通信/流依赖；保存timing.patch与前后源码哈希。若源码结构变化，补丁拒绝继续。

生产端到端取 `RBMD_TIMING_WINDOW_WALL_SECONDS / RBMD_TIMING_WINDOW_STEPS`；窗口首尾等待设备完成，取MPI最大墙钟。逐步非同步CPU span称 `Step-HostSubmission-Wall`，不能冒充GPU步耗时。LAMMPS取预热后最后一个Loop time。

两引擎生产运行取消launch blocking；分项诊断单独用 `--isolated`，两边HIP_LAUNCH_BLOCKING=1，RBMD设置 `execution.diagnostics.isolated_timing=true` 并启用串行GPU事件计时，LAMMPS显式设置 `timer full nosync`。两边均不为分项计时额外插入逐阶段MPI barrier；GPU同步和算法所需通信保留。诊断结果不参与正式加速比。正式实验前仍需人工审核模型、输入平衡及参数收敛。

两引擎保留Slurm的HIP/ROCR可见设备限制，按local rank选择设备；每个rank记录PCI地址，并检查同节点没有共用物理GPU。启动器隔离标准输入，顺序读取cases.txt时MPI不会吞掉后续case。
