# CVFF 并发修复的正确性验证

## G06 全盒 RBE 回归

主机检查：`python3 tests/run_rbe_full_box_host.py`。从生产文件提取并执行
MPI/非 MPI 采样分支，以 float/double 和 ASan/UBSan 检查完整盒、非复制子格
波矢和归一化 S；MPI 使用通信替身，不替代真实集成测试。

GPU 节点执行 `python3 tests/run_rbe_full_box.py --binary /path/to/rbmd --output /path/to/new-result --ranks 1`，
另用新目录测试 `--ranks 2`；非 MPI 编译用 `--ranks 0` 直接执行。
覆盖默认 MH 路径的 LJ、水（SHAKE）、PEO，未复制/复制输入、三次固定构型求力
及 Verlet/RBL 三步 NVT。MPI 构建还比较 `replicate`、`replicate_dims` 与同坐标显式扩胞的力。
非 MPI 默认 MH 尚无固定 seed 重放，故不做逐样本相等断言。
这是代码回归，不是 G07 精度实验或正式动力学数据；记录见
[验证报告](../diagnostics/g06_full_box_20261016/README.md)。

## α 调试开关回归

主机检查：`python3 -m unittest discover -s tests -p 'test_prepare_revision_matrix.py'`。
真实 GPU/MPI 检查：在计算节点加载编译/运行环境后执行
`python3 tests/run_debug_alpha.py --binary /absolute/path/to/rbmd --output /absolute/path/to/new-dir --ranks 1`，
再用另一输出目录和 `--ranks 2` 检查多 rank。每次输出目录必须不存在。
覆盖 RBE/EWALD 的自动、显式调试、输入拒绝及同 α 的力一致性；每个成功算例实际做三次固定构型求力，结果写入 `PASS.json`。不运行 NVE。

运行无需 GPU SDK 的算术测试：

```sh
python3 tests/run_cvff_bonded_host.py
```

脚本从当前 `cvff_op.hip.cu` 提取实际 Bond/Angle per-atom kernel，使用单线程
block 的主机替身编译执行，同时编译实际 GPU op 声明和参数转发。独立的标量
势能函数通过坐标和应变的有限差分验证力及六个 virial 分量；覆盖：

- float/double、多个相互作用在同一原子上累加、无拓扑原子；
- 稀疏 GID、owned/ghost 重排、跨分区键角、空 owned 分区；
- 周期映像、全局能量/拓扑计数、ghost 和 virial 缓冲的越界哨兵；
- AddressSanitizer 和 UndefinedBehaviorSanitizer。

这不是 GPU 并发或真实 MPI 测试，也没有验证整个 NPT 积分流程。

当前实现的依赖约束：

- `CVFF::Init` 一次性协商并发模式；任一 rank 禁用时，所有 rank 使用串行调度。
  初始化之后不再读取环境变量切换调度，避免 collective 次序在运行中分叉。
- RBL 开启能量时由完整列表完成 halo，RBL 使用 `BuildFromCurrentGhosts`。
  邻居缓存同时检查交换 epoch，不能仅凭原子数和 owned 位移判断 ghost 索引有效。
- 力诊断读取构表/迁移完成后的 owned 数量。
- per-atom Bond/Angle 分别给每个 owned 参与者写入 1/2、1/3 的 virial，
  使用 owned 数量作为六个分量的 pitch，与 `ReduceVirialOp` 一致。

2026-09-09 已完成的郑州 GPU/MPI 回归子集及数值、性能结果见
[并发验证报告](../docs/concurrency-overlap-validation.zh_CN.md)。以下是完整验收清单，
报告未覆盖的项目仍需单独验证：

1. 固定输入、随机种子、RBL sample epoch、精度和初态，比较原始基线、
   当前串行模式（`RBMD_FORCE_OVERLAP=0`）与并发模式；按 GID 比较各分力、
   总力、能量及 virial，不能只比较 thermo 总能量或当前的两个调度分支。
   Bond/Angle virial 应另与有限差分或独立参考比较，不能把旧版缺失值当正确基线。
2. 覆盖单卡/多 rank、Verlet/RBL、RBL 能量开关、fused/non-fused、skin 为零和非零、
   列表复用/重建、迁移与角/边/corner 周期转发。
3. 在各 rank 设置不同的并发开关，验证日志报告统一串行回退，且无 collective 挂起。
4. 让另一 builder 重建相同数量但不同 ghost 映射的 halo，确认旧缓存报告
   `exchange_epoch_changed` 并重建，随后两张列表使用相同的 ghost 布局。
5. 单步对照通过后，再验证 NVE 漂移、NPT 压力/体积和 SHAKE 约束残差，并运行设备检错。

## 百亿 EAM 扩展回归

```sh
python3 tests/run_billion_atom_host.py
python3 tests/run_eam_host.py
python3 tests/run_cvff_bonded_host.py
```

EAM Direct 的 GPU 基线对照需要 CUDA GPU：

```sh
CUDA_ARCH=sm_120 python3 tests/run_eam_direct_gpu.py
GPU_SANITIZER=memcheck CUDA_ARCH=sm_120 python3 tests/run_eam_direct_gpu.py
GPU_SANITIZER=racecheck CUDA_ARCH=sm_120 python3 tests/run_eam_direct_gpu.py
```

该测试从当前 `eam_op.hip.cu` 提取 Direct kernels，与保留旧逐原子、逐 stencil
遍历的独立 GPU 基线逐元素比较。它覆盖 float/double、32/64 位 ID、1/255/257/517
个 owned atoms、线性和非线性样条、融合 pair energy 开关、非整 block、不同 RBL
采样频率和上限。
默认架构为 `sm_120`，可通过 `CUDA_ARCH` 覆盖。

主机检查覆盖生产稀疏映射、分区晶胞复制、模拟 MPI 读入协议、64 位消息分块，
以及 EAM 内核/接口和能量归约算术。模拟 MPI 和 block shim 不替代 GPU/MPI 集成测试。
构建、32000 原子 smoke 和百亿 Cu 示例、集群验收及路径限制见
[百亿 EAM 说明](../docs/billion-atom-eam.zh_CN.md)。

## OPLS LJ 混合回归

```sh
python3 tests/run_lj_mixing_host.py
```

从生产代码提取 LJ 力表构造、RBL 能量、SOG 力/能量 kernel 和 op 参数转发，
在 float/double 下使用 ASan/UBSan 执行。独立势能及有限差分检查不同类型参数、
同类型参数、几何/算术混合、截断外的零作用、普通原子对，以及
`special_bonds: [0, 0, 0.5]` 的 1–2/1–3 排除和 1–4 半权重。
库仑使用零电荷替身；本测试不覆盖 GPU 调度、MPI 或带电体系的库仑计算。

## MPI image 与 unwrapped 回归

在已分配 GPU 的环境中，对 1、2、8 rank 分别运行（每次使用新输出目录）：

```sh
python3 tests/run_mpi_images.py --binary /path/to/rbmd --output /path/to/new-result --ranks 8 --lammps /path/to/lmp
```

覆盖 atomic、charge、full 键角拓扑和 full+SHAKE，各运行正反方向两组 NVT，
每组 500 步。初始 image 非零，每个方向总位移超过一倍盒长；多 rank 必须记录
真实迁移，选定字段的轨迹还检查所有权变化。普通轨迹与选定字段轨迹按 ID 同
独立连续重建比较；可选的 LAMMPS `rerun + compute msd` 读取原生坐标生成的
image，与独立 Python MSD 比较。该夹具不用于物理精度或性能结论，不测试 NVE。
