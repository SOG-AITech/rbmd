# 小体系第一轮：LAMMPS 原生后处理

2026-10-20，运行目录：
`NewBenchmark/runs/04-staged-20261020/postprocess-round1`。

| 来源轨迹 | 作业 |
|---|---|
| LJ / RBMD | 947818 |
| LJ / 独立 LAMMPS PPPM | 947819 |
| 水 / RBMD | 947820 |
| 水 / 独立 LAMMPS PPPM | 947821 |

短回放作业 947803：四套轨迹各 3 帧，全部通过；正式作业等待完整性检查
947730 结束后启动。检查失败会阻止对应计算；仅展开镜像异常允许计算不依赖
跨帧展开坐标的量，并明确拒绝将原始 MSD 拟合结果作为有效扩散系数。

## 方法与输出

- `prepare_native.py` 生成 LAMMPS 输入；原样读取单文件或四分片轨迹，不生成完整合并副本。
- LAMMPS `compute rdf` 计算 1–1、1–2、2–2 RDF，bin 宽 0.05；LJ 半径 16，
  水半径 9。水的分子内 1–2/1–3 对排除，与生产时 special_bonds 一致。
- `compute msd com yes average no`：初始帧为参考，LJ 两类离子分别计算，
  水用氧原子。拟合 50000–100000 步，并列出两个半窗口斜率供稳定性检查。
- Szz 使用 LAMMPS 原子变量和 `compute reduce`，三个轴各前四个波矢，
  归一化为 `|sum(q exp(ikr))|² / sum(q²)`。
- 水的中性 OHH 分子用 molecule chunk 找到自己的氧，在每帧重建唯一的短 O–H
  相对向量，LAMMPS 求总偶极矩；逐帧验证 O–H 长度和每分子一个氧。
  在 LAMMPS 来源轨迹上同时与内置 `compute dipole` 比较，最大差需小于 1e-6 eÅ。
  这只解决分子内部偶极矩，不推断跨时间的扩散路径。

`pair_style zero` 和零键角势仅建立分析所需邻居关系，不积分、不重算 PPPM。
两套来源仍分别是 RBMD 和独立 LAMMPS PPPM 动力学。

`reduce_native.py` 只归约 LAMMPS 输出：20000–100000 步的 RDF 均值、Szz 均值、
偶极矩方差和线性拟合。介电常数使用同窗口原始动力学的实测平均温度。
输出 `results.json`、`rdf_mean.csv` 和每帧全局量 `observables.csv`；
无逐原子 CSV，不删除原始轨迹。

## 当前限制

RBMD 的 LJ 和水展开坐标分别出现约 271 和 90 Å 的盒长跳变。
原始 MSD 保留作诊断；`D` 设为 null，状态为 `INVALID_UNWRAPPED_IMAGES`。
水的偶极矩仅在完整分子几何检查和交叉验证通过后报告。

这仍是三轮中的第一轮。前 20% 截去是声明的分析窗口，不能证明平衡或收敛；
水总长仅 50 ps。四段介电估计用于观察漂移，不是独立重复的置信区间。
第二、三轮、中大体系和 PEO 均不由这组后处理作业启动。

实现依据：[LAMMPS rerun](https://docs.lammps.org/rerun.html)、
[read_dump](https://docs.lammps.org/read_dump.html)、
[compute msd](https://docs.lammps.org/compute_msd.html)、
[reduce/chunk](https://docs.lammps.org/compute_reduce_chunk.html)。
