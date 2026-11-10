# 08 正式输入、独立起点与审计

本节只准备候选态和证据，不把数据头的timestep当成平衡证明。所有命令在NewBenchmark目录执行。

## 两类输入分开准备

| 用途 | 需要的构型 | 重复什么 |
|---|---|---|
| **03 MJ2(a)：力误差和方差随N变化** | **每个N一个弛豫构型即可** | 冻结同一构型的位置，反复随机抽批；不需要五个独立起点 |
| **04：物性统计误差条** | 在所选体系、同一个N和相同物理条件下，默认准备5份独立弛豫起点 | 分别跑5条动力学轨迹，各得一个物性估计，再统计均值和误差条 |

**本节的“五个平衡候选态”只用于04物性误差条，不是给MJ2(a)的每个N准备5份。** 数字5是当前统计流程选定的重复次数，不是审稿人规定的构型数量。单次RDF可以从一个弛豫构型运行；03也不依赖下面的五态任务完成。

## 归档水基础输入

```bash
python3 08_inputs/prepare_water_base.py --data-root ../rbmd/benchmark/data --output inputs/water-base
```

原2703原子h2o_rdf.data缺少系数段；脚本从归档h2o.data补入SPC/E的Pair/Bond/Angle Coeffs，检查原子数和质量表，保留原坐标/速度/拓扑，记录两份源文件和结果的SHA256。不会覆盖已存在输出。

## 仅用于04物性误差条：五个独立起点的候选态

下面的命令针对所选物性体系的一组独立轨迹，不按03的各N批量执行。生成任务不表示已经得到5份合格平衡态；不同随机速度也不自动保证后续构型相互独立。

```bash
python3 08_inputs/prepare_states.py --system systems/water.rdf.json --steps 2000000 --output runs/water-equilibration-v1
# 提交前自行按体系弛豫时间确定单次平衡长度
while IFS= read -r case; do sbatch common/job.slurm "$case"; done < runs/water-equilibration-v1/cases.txt
```

每个任务从同一构型重新生成不同随机种子的速度，运行NVT并输出state.data/state.restart。默认示例2,000,000步是1 ns，不自动证明介电涨落已去相关。检查temp/pe/press/density时间序列、结构和相关时间；必要时继续平衡。states.json中的候选状态必须配合实际日志审核，不能直接改标签冒充验证。五条物性轨迹各自拟合后汇总置信区间；计时五次重复与这五个状态无关。


## 历史复制与全盒源码

```bash
mkdir -p validation
python3 08_inputs/audit_sources.py --code-root ../rbmd --output validation/source-input-audit.json
python3 08_inputs/check_replica_correlation.py /path/full-box.dump --replicate 3 3 3 --output validation/replica-correlation.json
```

审计遍历benchmark的JSON、LAMMPS输入/日志，记录read_data与replicate行和源码哈希。JSONC注释允许读取；模板未替换、include/变量和缺失运行日志会保持未确认，配置本身不等于执行过。图中旧运行是否用复制应逐条引用这些记录及原日志。

副本检查要求LAMMPS连续ID和标准复制ID偏移，比较同源原子折回基础胞后的位置。近零说明仍有重复位置；非零不能单独证明统计独立。其他ID映射必须先明确映射，工具不会猜测。W0的来源日志现已找到：0.9 ns、100 fs阻尼的全盒松弛，有限观测量检查通过；原判据明确不证明所有复制相关性消失，不能直接认证本轮5 fs RDF的五个独立态。摘要见 `evidence/20261017/W0-provenance-summary.json`。

源码审计记录全盒采样关键函数和相关文件SHA256、Git提交及工作区变动。未提交改动也计入证据，但源码存在不等于现有二进制包含它；需记录独立构建及短测试。

## 参数表

```bash
python3 08_inputs/report_parameters.py --systems systems --output report_parameters.csv
python3 08_inputs/report_parameters.py --cases runs/lj-force-v1 --output runs/lj-force-v1/parameters.csv
```

第一条可填原子数、data哈希、电荷、LJ Bjerrum长度及配置；第二条补实际α、PPPM网格/order/gewald、MPI版本和实际RBL样本统计。未运行的字段写NOT_MEASURED，不填名义p代替实际p。LJ公式lB/σ=1/(εr T*)采用单位约化电荷；数据中实际q及Σq²同时保留。

RBL统计需要新RBMD构建的验力插桩：按owned atoms及采样时刻加权，验证所有rank和采样完整，输出均值/标准差/最小/最大。计时路径不读取这些数组。
