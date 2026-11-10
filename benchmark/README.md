# NewbenchmarkOrigin — 无结果脚本包

整理于2026-11-03。原 NewBenchmark 保持不变。本目录不含运行结果、轨迹、二进制或输入大数据；没有提交作业。

## 使用范围

- 00–04、06–08：通用预检、验力、物性和输入处理脚本。根 systems/water.json 与 lj.json 对齐04实际参数。默认运行步数1000；固定构型验力的samples为独立统计量，不是MD步数。04的1000步只用于短程检查，不能替代此前长轨迹物性结果。历史后处理脚本中的采样窗口有各自限制，使用前按新轨迹长度设置。
- 05：当前选定规模的线性、强扩展、弱扩展；水使用10月27日PairOpt V1，LJ保留当前结果对应的原构建。每节点4卡，每卡1 MPI进程；同步Pair+Kspace计时，保留段内通信，不含单列Neighbor等成本。
- 05配置只在 `05_timing_scaling/matrix.json` 维护；旧矩阵、旧结果和自动提交链未复制。

## 05 生成输入（不提交）

在目标服务器复制本目录后，从该目录执行：

```bash
python3 05_timing_scaling/prepare.py --system all --output runs/scaling-1000
```

生成84个独立作业输入：水14点×2次×2引擎，LJ14点×1次×2引擎。强弱4卡基准按网格分别使用，不混用。生成器只写文件，绝不调用sbatch。生成脚本含绝对路径，目录搬迁后请重新生成。

运行步数设置为1000、无额外预热。**旧RBMD循环会产生1001次力计算；LAMMPS执行1000步。** 原生RBMD阶段平均值不可再次除以1000；LAMMPS总计时按其日志Loop步数归一化。增加步数不解决分段边界或等力误差可比性问题。

水：cut10/core6.5 Å，p100/P300，skin2 Å，alpha0.065776184，298 K，Tdamp10 fs，dt0.5 fs。原盒子扩胞采样；强扩展固定7,881,948原子，弱扩展每节点7,881,948原子。
LJ：cut16/core8，p30/P200，skin0.3，alpha0.024285356，T1，Tdamp0.1，dt0.001；强扩展21,000,000原子，弱扩展每节点21,000,000原子。

## 外部依赖

- 水性能二进制：`/public/home/acymkxdx47/pair_opt_scaling_20261027/software/rbmd`，SHA256 `fd02e2e589e29e75bc1235028bc56b27813a49addecf77c7acdd4d59225b8c8c`。
- 水网格/屏障包装器：同目录 `runtime/mpi_probe.so`，按matrix中的哈希核对；包含RBL Allreduce前屏障。
- LJ性能二进制：`/public/home/acymkxdx47/rbmd/build/rbmd`，按matrix固定哈希核对；不使用水PairOpt构建。
- LAMMPS：`/public/home/acymkxdx47/software/rbm26/build-hip-single/lmp`，固定单精度构建与哈希。
- 数据路径保留实际初始构型路径，包括原NewBenchmark运行目录内的initial-rep1.data。它们是必需外部输入，不随本脚本包复制；不可删除后仍期待本包直接运行。
- 正确性步骤默认用 `ReviewRBMD/rbmd/build/rbmd`，不自动换成性能版本。软件环境路径见env.sh。

检查范围：Python语法、shell语法、全部84份05输入离线生成、步数及4卡/节点布局；未远程核验当前二进制/数据是否仍存在，未运行MD。执行前运行器会核对性能二进制和包装器哈希。
