# 体系输入与力场

正式输入位于郑州 `ReviewRBMD/rbmd/benchmark/data`，本包所用归档文件的SHA256已与该目录README核对。数据头的timestep和归档描述只是来源证据，不能独自证明当前力场下已平衡。

| 配置 | 数据 | 原子数 | 设置 |
|---|---|---|---|
| lj.json | lj_salt.data | 200000 | LJ单位，q=±1，T*=1，dielectric=1，因此单位电荷Bjerrum长度lB/σ=1 |
| water.json / water.rdf.json | h2o.data | 72981 | SPC/E，298 K，SHAKE；普通阻尼100 fs，RDF阻尼5 fs |
| water.base.json | NewBenchmark/inputs/water-base/water2703.data | 2703 | 用08脚本给h2o_rdf.data补归档系数；不改变坐标，3×3×3后为72981 |


平衡记录至少包括：构建来源、组成/端基、盒子/密度、净电荷、复制和去相关方式、最小化及平衡历史、温度/压力/阻尼、平衡判据、最终data哈希。08能生成五个独立速度种子的平衡候选任务；只有核验平稳性和去相关后才能把其输出作为 `--states`。所有配置目前保留UNVERIFIED标签。

准确性计算读取同一份显式全盒data，避免暗含复制。05计时可用 `--replicate`，两引擎同步复制。02是专门的复制表示正确性实验，不把相同副本当成独立平衡大盒。

水和 LJ-salt 两引擎均采用 LB 混合规则（sigma 算术、epsilon 几何）。
