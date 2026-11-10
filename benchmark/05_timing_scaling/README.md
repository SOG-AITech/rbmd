# 05 当前有效配置

以matrix.json为唯一配置入口；使用prepare.py生成，不自动提交。详见[根目录说明](../README.md)。

水采用PairOpt V1，LJ沿用原构建，均1000步。生成每例job.slurm；不使用固定单节点通用job.slurm提交多节点任务。原历史结果未随包复制。
