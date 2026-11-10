# 01 固定构型 RMS 与参数校准

最新结果见 [RESULT/01](../RESULT/01/RESULTv2.md)。

1. 用同一个显式、已平衡 data 生成扫描：

```bash
bash 01_force_accuracy/prepare.sh --system systems/lj.json --output runs/lj-force-v1 \
  --cutoffs 8 12 16 --P 100 200 400 800 --p 30 60 120 --samples 1000 --kmax 12 12 12
```

2. 先运行每个 rs 下的 `pppm_0.0001`、`pppm_1e-06`、`pppm_1e-07`；再运行 `vl_ewald`、`rbe`、`rbl`、`combined`。每个 case 都有 `run.sh`；`cases.txt` 给出完整清单。
3. 在同一 rs 下另增大 kmax，直到 Verlet+Ewald 相对更精细参考稳定，才把 Ewald 视为已收敛。仅给出 kmax 值不能证明收敛。
4. 分析：

```bash
python3 01_force_accuracy/analyze.py runs/lj-force-v1 \
  --reference-id-offset -1 --reference-position-tolerance 1e-6 --output runs/lj-force-v1/summary.json
```

当前 RBMD 力 CSV 是原生 0 起始 ID，LAMMPS dump 为 1 起始，故示例显式用 -1。轨迹输出 ID 已加 1，不要对轨迹再减 1。脚本检查 ID、固定坐标、样本数、有限值；若实际编号不同，应查清后更改参数，不放宽坐标容差掩盖错误。

## RMS 怎样得到

对同一构型的 N 个原子重复 M 次求力，参考力 Fref 来自 PPPM 1e-7（先确认 1e-6→1e-7 变化足够小）：

- RMS = √[ΣmΣi |Fmi−Fref,i|²/(MN)]，这里是三维向量 RMS，不再除 3。
- relative RMS = RMS / √[Σi |Fref,i|²/N]。
- bias = √[Σi |mean_m(Fmi)−Fref,i|²/N]。
- population variance = ΣmΣi |Fmi−mean_m(Fmi)|²/(MN)，并检查 RMS²=bias²+variance。
- 无偏方差改用 M−1。不能用 bias 代替瞬时 RMS，也不能先平均随机力再宣称每步误差达标。

四组保留完整物理力：Verlet+Ewald（确定性候选）、Verlet+RBE（只改变长程算法）、RBL+Ewald（只改变短程算法）、RBL+RBE（联合）。不是删除另一部分相互作用后比较总力。

参考力用双精度 CPU LAMMPS `run 0`，通过 `00_preflight/reference.slurm` 提交。对每个被测点，分析器检查两份参考力（PPPM1e-6与1e-7）的绝对RMS差 ≤0.1×该点测得的绝对RMS误差。超限保留数值并标为 `REFERENCE_LIMITED`，不得声称参考已收敛；继续提高参考精度验证。PPPM收敛检查仍不能代替确定性Ewald的kmax收敛检查。只报告实际 RMS、bias、variance 和参考收敛状态，不再生成 equal_accuracy_candidate，也不用于筛选扩展性计时。零参考力不生成相对误差。

先选 P，再调 p/rc；生成器默认 rc=rs/2。要专门扫描 rc，复制 system JSON/生成输入并保持所有对照一致。当前具备 RBE 的 P 扫描、RBL 的 p 扫描和联合 P×p 扫描。不同 rs 改变自动 α；这不是独立 α 扫描，不能将 cutoff 效应全部归因于 α。正常运行仍自动选 α；单独的 α 参数扫描显式标记 `alpha_scan=true`，低模态变体仍禁用。记录运行日志的实际 α、实际壳层采样数，名义 p 不等于实际 p。

可单独使用 `common/analyze_force_samples.py` 分析任意一次完整的 rank 分片；参考力也可来自验证收敛的 Ewald CSV。内存 O(N)；仅用于本次RDF准确性体系，不安排最大N验力。

## 自动生成 RMS–α 实验

普通 RDF 仍由 RBMD 自动选 α。参数实验先读取一次相同体系普通运行的 `stdout.log` 中实际自动 α，以它为中心自动生成 0.5、0.75、1、1.5、2 倍的 α；不需要逐份手改 JSON。

```bash
bash 01_force_accuracy/prepare.sh --system systems/lj.json --output runs/lj-alpha-v1 \
  --cutoffs 16 --P 200 --p 30 --samples 1000 --kmax 12 12 12 \
  --alpha-log runs/lj-force-v1/rs16/combined_P200_p30/stdout.log
# 在 Slurm 上逐一提交生成的算例；提交后等全部完成再分析。
while IFS= read -r case; do sbatch common/job.slurm "$case"; done < runs/lj-alpha-v1/cases.txt
python3 01_force_accuracy/analyze.py runs/lj-alpha-v1 --reference-id-offset -1 \
  --output runs/lj-alpha-v1/summary.json
```

`--alpha-log` 只允许 force 模式和一个 cutoff；默认五个倍率可用 `--alpha-factors` 改。示例 P、p 各一个值，因此每条曲线固定 P、p、rs、rc、构型，只变 α。扫描包含 RBE、RBL、联合和确定性 Ewald 对照。不是让正常 RDF 自适应改变 α。

输出 `summary.json` 和同名 `summary.csv`，自动列出实际 α、α 来源、P、p、rs、rc、RMS、相对 RMS、bias 和参考收敛状态。绘制同一算法、同一 P/p/rs/rc 的 RMS 对 α 即可；不同参数不能连成一条“仅随 α”曲线。RBMD α 对应 LAMMPS gewald²，单位为长度⁻²。参考 PPPM 使用自己的自动参数，不强制跟着扫描 α。

倍率只是初始采样范围，不代表最佳值或已收敛。Ewald 的 kmax 必须在扫描范围内分别检查，尤其最大 α 端；固定 kmax 不足的误差不能冒充随机采样误差。实际日志 α 必须与请求值相符，否则分析器报错。

坐标比较对参考全盒采用最小映像。默认参考容差为 `1e-6 + 2^-22 × max(|boxlo|, |boxhi|, L, 1)`，覆盖单精度舍入；同一RBMD固定构型的不同采样仍以1e-10严格检查移动。坐标量化导致的力误差仍包含在结果中，不能把空间容差理解为抹去力误差。

RBL实际采样数由独立计时源码副本中的accuracy-only插桩输出 `revision.rbl_samples.rank*.csv`；统计的是owned atoms实际壳层列表长度，不是名义p，包含随后按special-bonds缩放/排除的已选对。生产计时不执行这段设备回读。使用08的参数汇总脚本输出均值、标准差和极值。
