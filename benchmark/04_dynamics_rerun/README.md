# 04 按体系分开的 LAMMPS 后处理

最新结果见 [RESULT/04](../RESULT/04/RESULT.md)。

| 体系 | 本次指标 | 入口 | 是否需要另跑轨迹 |
|---|---|---|---|
| 水 | 静态介电常数 εr | `water_dielectric/README.md` | 可复用符合条件的 RDF 轨迹 |
| 带电 LJ | 小 k 电荷结构因子 Szz(k) | `lj_structure/README.md` | 可复用符合条件的 RDF 轨迹 |
| 两种体系 | RDF | 本页下方 | 保留原 RDF 任务 |

LAMMPS 回放负责总偶极矩、各 k 的电荷傅里叶分量；Python 计算偶极涨落公式、汇总均值和误差条。水与 LJ 的专门回放不再计算 MSD。两个专门入口各有脚本和 README；不做 NVE 漂移、恒温器热通量及 5 ns/50 ns 时长比较。

## 轨迹与统一约定

普通 RDF 轨迹都由 RBMD 自动选 α。水 RDF 保留5 fs阻尼、0.5 fs时间步、链长3、子循环1；同一对照的两个引擎使用相同 data、模型、约束和温控参数。轨迹运行不参与扩展性计时。

生成器会拒绝 NVE 输入，以及水 非5 fs阻尼的RDF输入；请使用 `*.rdf.json`，不要把100 fs阻尼的计时模板传入本节。`--steps` 只指定一次生产长度，不自动生成5 ns/50 ns对照。

```bash
# RDF 轨迹示例；LJ 换成对应 system JSON。
bash 04_dynamics_rerun/prepare.sh --system systems/water.rdf.json --output runs/water-rdf-v2 \
  --steps 100000 --stride 100 --repeats 5 \
  --states /data/s1.data /data/s2.data /data/s3.data /data/s4.data /data/s5.data
# 将同一次 RBMD 运行的所有 rank 分片合并。
python3 common/merge_trajectory_shards.py -o /work/merged.dump /work/run/rbmd.rank*.trj
```

单次RDF用一个弛豫构型作为起点就可以；五份独立平衡起点用于这里的跨轨迹统计误差条，不是运行一次RDF的前提，也不是03 MJ2(a)的要求。做五次统计时，两引擎逐次配对使用相同起点。未给 `--states` 时只是同一起点重复，不能当独立平衡态误差条。帧之间相关，不能把每帧当独立样本。

回放输入须含 id/type/q、x/y/z、xu/yu/zu、vx/vy/vz。转换器检查原 data 的原子 ID、类型、电荷，以及轨迹各帧完整性；从显式展开坐标恢复映像，不从相邻帧最小位移猜跨界。限定固定正交盒；缺帧、变盒或 triclinic 会拒绝。水介电回放额外检查每个三点中性水分子的展开坐标一致性，避免跨界造成伪偶极涨落。

每个专门回放同时用于原始 RBMD 和原始 LAMMPS 轨迹；相同窗口和采样间隔。`--first` 必须是实际帧且为 stride 整数倍。LAMMPS 从原 data 读取电荷/拓扑，用 dump 更新坐标、速度、映像；没有时间积分、恒温器或 SHAKE 约束反力。当前仍保留原力场计算，回放速度不作为 MD 性能结果。

## RDF 回放

```bash
python3 04_dynamics_rerun/prepare_rerun.py --observable rdf --system systems/water.rdf.json \
  --case /work/run --trajectory /work/merged.dump --output /work/rdf-rerun \
  --first 1000 --stride 100 --radius 8 --dr 0.1
sbatch common/job.slurm /work/rdf-rerun/lammps
python3 04_dynamics_rerun/compare_rdf.py rbmd-rdf.dat pppm-rdf.dat \
  --peak-min 2 --peak-max 4 --output rdf-comparison.json
```

RDF 默认保持同一分子内 special_bonds 排除约定。比较要求一致窗口/分箱，输出第一峰偏差、峰位偏移和曲线 RMS；5%只检验指定峰窗口。参考力统一走01的双精度CPU流程，04不附带单精度参考力输出。

## 分析与五次汇总

各指标的 `analyze.py` 从对应 `postprocess.json` 读取体系、单位、dt 和指标，避免给水/LJ强行套扩散拟合。`--first/--last` 选择实际有帧的分析窗口。输出 JSON 及同名 CSV：介电保留逐帧数据，Szz保留各k与均值。

```bash
python3 04_dynamics_rerun/aggregate.py r1.json r2.json r3.json r4.json r5.json \
  --independent-states --output errorbars.json
```

按指标汇总 εr 或每个同一k的Szz，输出 mean、SD、SEM 和 t(4) 95%区间。不混合不同体系/单位/原子数；Szz不混合不同盒长产生的不同k。每个独立轨迹先得一个估计，再跨轨迹统计。


**介电汇总区分目标温度与实测均温。** `prepare_rerun.py` 从原算例的恒温器设置记录 `target_temperature_K`，拒绝升降温目标；`analyze.py --temperature` 仍填写原始受约束轨迹在分析窗口内的实测均温，每条轨迹用自己的温度计算 εr，不改写温度、不重标度 εr。五份结果必须具有相同的目标温度（相对容差1e-8仅用于目标值舍入），实测均温允许不同。

汇总默认要求每条实测均温距共同目标不超过1 K，可用 `--temperature-tolerance` 明确调整。1 K是可审查的流程阈值，不是物理等价性或收敛判据；超差时先检查温控和分析窗口，不应只为通过检查而增大阈值。298.00–298.04 K在共同目标298 K下可正常汇总，目标298 K和310 K的结果仍拒绝混合。输出保留目标值、五个实测均温、温度统计、所用阈值和 εr 的 mean/SD/SEM/95% CI。

旧的 `postprocess.json` 没有目标字段时，重新分析并用 `--target-temperature 298` 指明原恒温器目标；有记录时不允许用此参数覆盖不同目标。旧单轨迹结果缺目标字段会拒绝汇总，须重新分析，不能把实测均温冒充目标。此流程不认证轨迹独立性或偶极涨落收敛。

官方依据：[LAMMPS rerun](https://docs.lammps.org/rerun.html)、[MSD](https://docs.lammps.org/compute_msd.html)、[dipole](https://docs.lammps.org/compute_dipole.html)。
