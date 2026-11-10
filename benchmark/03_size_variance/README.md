# 03 MJ2(a)：固定参数下力误差和方差随 N 的变化

最新结果见 [RESULT/03](../RESULT/03/RESULTv2.md)。


2026-10-20：03 改用 RBMD 内置 RMS 与在线偏差/方差。按本轮用户要求，验证通过后运行 LJ 和水，排除 PEO；每个体系三个规模、每种方法1000批。

同一体系、同一密度下，每个 N 使用一个已选定的弛豫构型；固定 P、p、α、rs、rc 和 dt，冻结位置重复抽样。当前只报告 RBE、RBE+RBL，两种方法分别成曲线；单独 RBL+Ewald 暂缓，不生成也不要求其结果。RBMD 单精度；参考力仅用 CPU 双精度 LAMMPS PPPM run 0。

**每个 N 一个弛豫构型即可，不要求五个独立平衡态，也不用先为每个 N 跑 RDF。** 这里的重复是同一冻结构型上的随机抽批；`--samples` 是批次数，不是构型数或 MD 步数。04用于物性误差条的五条独立轨迹不属于本项。

## 已核对的输入入口

根目录：`/public/home/acymkxdx47/ReviewRBMD/revision_lj_water_20261015/relaxed_systems_20261017`

| 体系 | N | 根目录内入口 | 当前范围 |
|---|---:|---|---|
| LJ L0 | 200000 | `L0_small_LJ/test_input.data` | 纳入 |
| LJ L1 | 2000000 | `L1_medium_LJ/test_input.data` | 纳入 |
| LJ L2 | 28000000 | `L2_large_LJ/test_input.data` | 纳入 |
| 水 W0 | 72981 | `W0_small_water/terminal.data` | 纳入 |
| 水 W1 | 1970487 | `W1_medium_water/terminal.data` | 纳入；210000步制备状态 |
| 水 W2 | 15763896 | `W2_large_water/latest_validated.data` | 已采用作测试源：USER_ADOPTED_VALIDATED_180000_SP_STEPS；仍须审核 |

LJ 原始 terminal.data 不含 Pair Coeffs，使用目录中已登记的 test_input.data 符号链接（snapshot_with_coeffs.data）。两个引擎读取同一文件。水直接使用含系数的原始 terminal.data，不套用旧的速度转换表示。不会修改这些源文件、复制扩胞或把目录名当成平衡证明。

生成时读取各目录 manifest.json，核对体系、N、系数和密度，记录来源元数据、解析后的真实路径及文件 SHA256；制备中或非最终态拒绝生成。W2 如纳入本次比较，须核对当前 manifest 后明确加入 --data。归档的“采用作测试源”不等于已经证明统计平衡；正式解释前仍需人工审核制备依据。

## 仅生成输入

以下为输入生成接口。P、p、α、截断必须明确设置；03不再接受 `--kmax`，因为本轮没有 Ewald 分支。示例数值只是语法示例，不表示已经收敛。

```bash
cd /public/home/acymkxdx47/ReviewRBMD/NewBenchmark
bash 03_size_variance/prepare.sh --system systems/lj.json \
  --data L0_small_LJ/test_input.data L1_medium_LJ/test_input.data L2_large_LJ/test_input.data \
  --P 200 --p 30 --alpha 0.024 --rs 16 --rc 8 \
  --samples 1000 --output runs/lj-size-v1

bash 03_size_variance/prepare.sh --system systems/water.rdf.json \
  --data W0_small_water/terminal.data W1_medium_water/terminal.data \
  --P 200 --p 30 --alpha 0.08 --rs 9 --rc 5.5 \
  --samples 1000 --output runs/water-size-v1
```

每个 N 生成三个 CPU 参考（PPPM 1e-4/1e-6/1e-7）和两个 RBMD 固定构型任务（RBE、RBE+RBL），共五个算例。`cases.txt` 仅列路径，生成器不调用 sbatch 或 mpirun。后续参考用 `00_preflight/reference.slurm`，RBMD 用 `common/job.slurm`，节点数和内存人工确定。

α 通过 debug_alpha 明确锁定，并由实际日志核验；普通 RDF 自动 α 不变。固定构型采样不推进 MD，不使用恒温器反力，不输出热通量、能量漂移或动力学轨迹。水的参考力是保守力，不含 SHAKE 约束反力，与01相同。

本轮暂停单独 RBL+Ewald，Ewald 的并行可行性和 kmax 收敛不作为当前03的前置条件。RBE 和 RBE+RBL 均使用随机倒空间采样。参考统一使用 LAMMPS PPPM；1e-7 并非自动合格，仍沿用01比较1e-6与1e-7参考差异，参考不足时03拒绝汇总。论文须明确参考算法及其数值收敛依据。本轮结果不能声称已完成单独RBL的误差隔离。

## 后续分析（完成运行后）

```bash
python3 03_size_variance/analyze_online.py runs/lj-size-v1
python3 03_size_variance/collect.py runs/lj-size-v1 --output runs/lj-size-v1/size-statistics.json
```

水使用相同流程替换根目录。使用新目录生成本轮两分支任务；既有目录和历史结果不改动。保留01的周期折叠/单精度坐标容差与参考收敛检查；03再核对每点真实N、源文件哈希、完成状态、固定参数、实际α和样本数。旧的手填 statistics/alpha CSV 接口已替换为直接读取运行证据。

对 N 个原子、M 个批次，输出：

- `mse = (1/NM) Σ|Fᵢᵇ−Fᵢref|²`，即这里估计的 E|χᵢ|²；同时输出 RMS=√mse。
- `rms_bias`：批次平均力相对参考的偏差。
- `variance_population`：围绕批次平均力的涨落；mse=偏差平方+总体方差。
- `variance_unbiased`：分母 M−1 的无偏方差；另报方差/N、mse/N 和 log(方差) 对 log(N) 的斜率。

这些是单构型上的随机批统计，不是五个独立平衡态的不确定性。只解释已测 N，不能外推最大 N 准确性。L2 是非立方盒，盒形变化也须在讨论中披露。

## 仍然不做

最大 N 的恒温器热通量、短 NVE 漂移继续排除；不自动生成全套最大 N 验证、每个扩展性点的物性、等 RMS 性能比较。05 只负责性能。

03 的 `forces=false`，不输出逐原子 CSV。每个随机任务仅输出1000行 `revision.force_errors.csv` 和最终 `revision.force_statistics.json`。前者保留逐批RMS和累计RMS，并增加累计偏差、总体方差、无偏方差；第一批的无偏方差留空。后者供汇总器使用。

参考输入必须为按 ID 严格递增的正交周期盒 LAMMPS dump（`modify sort id`）。各 MPI rank 顺序读取一次参考文件，仅保留本地 owned 原子的参考值、固定位置和双精度 Welford 统计；用16384项的小块校验全局原子归属，不广播整文件、不存全局N项归属数组，取消2GB广播限制。内存随本地原子数增长，首次文件读取的I/O仍随rank数增长。逐批允许原子重排序，但拒绝坐标移动、ID重复/缺失和ownership变化。

`analyze_online.py` 用流式逐行比较三个PPPM参考，不读入完整帧；检查运行回执、实际α、样本数、参考归一化，以及 `mse = rms_bias² + variance_population`。参考差异仍须不超过被测RMS的10%；`collect.py` 继续拒绝不收敛参考。

本轮已验证二进制统一部署到 `/public/home/acymkxdx47/ReviewRBMD/rbmd/build/rbmd`，全局RBL发布登记已同步。原隔离 build 已删除；历史回执保留原路径用于追溯。验证证据：`validation/online03-20261020`；算例：`runs/03-online-20261020`。源输入的平衡认证状态仍以manifest为准。



### PPPM参考内存（2026-10-20）

03 新生成的参考使用 `kspace_modify order 7`，精度目标仍为1e-4/1e-6/1e-7，
保持CPU双精度和物理截断。原order=5在中体系1e-7形成过大FFT网格，曾触发48GB配额OOM；
本轮已完成的order=5参考保留，未完成参考在新目录记录失败证据后改用order=7。
最终记录实际grid/order/gewald；是否足够精确由参考收敛对照决定，不以阶数替代验证。
