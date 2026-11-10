# 07 稀 LJ 电解质的 PPPM 调优（截图 MN3(d)）

这一条与“不做同 RMS 成本对比”的 MN3(g) 分开。截图质疑稀体系的大体积使 PPPM 网格昂贵，90–180×不能外推到所有密度；还要求给 PPPM 调整实空间/倒空间工作量的机会。

默认100步预热＋2000步计时，预热另计；两个引擎采用相同计时窗口。

## 自动截断扫描

```bash
bash 07_pppm_tuning/prepare.sh --system systems/lj.json --output runs/lj-cutoff-4gpu-v1 \
  --cutoffs 8 12 16 --repeats 5
while IFS= read -r case; do sbatch common/job.slurm "$case"; done < runs/lj-cutoff-4gpu-v1/cases.txt
# 等完整清单全部完成后：
python3 07_pppm_tuning/summarize.py runs/lj-cutoff-4gpu-v1 --output runs/lj-cutoff-4gpu-v1/summary.json
```

LAMMPS 使用 `pair_style lj/cut/coul/long LJ_cut Coul_cut`，**LJ 截断固定为 system JSON 的值，只扫描第二个库仑实空间截断**。PPPM 目标精度1e-4、order5，gewald/grid自动确定；密度、data、温度、时间步、精度、节点布局保持一致。RBMD 的 P/p/rc/rs 固定，每个重复只运行一次基线，不随 PPPM cutoff 变化。这样避免调优时无意改变 LJ 物理模型。

汇总脚本要求全部算例成功、同一 data SHA256 和布局、正式计时、每点重复数一致；从日志提取实际 gewald/grid/order。各 cutoff 先取耗时中位数，再选已测点中最快 PPPM，报告它相对固定 RBMD 的加速比；保留全部点。没有强制同 RMS 筛选，也不声称同 RMS。order/mesh未穷尽，不能把“已测点最快”写成“全局最优”。自动 α 扫描仅在01做，本目录不引入准确性重复实验。

## 4 到 64 DCU

每个布局生成独立 output，输入 data 完全相同；64 DCU 用16节点×4卡。分别报告两个程序的强扩展效率 `t(4)*4/[t(64)*64]`，以相同参数的点计算；不要混用各布局不同最优cutoff的时间声称固定参数强扩展。

加速比从169降到90本身不能单独确定RBMD的扩展效率，需同时给出LAMMPS和RBMD原始耗时。截图中的55%需用实际记录核对，不直接写进结论。

## 高密度体系与答辩边界

高密度约0.1的对照需要另建并平衡；不能把旧盒缩小后立即计时。具备该data后，同一命令增加 `--data /absolute/dense_equilibrated.data`，两个程序读取同一份文件，并使用单独output。

当前没有在本目录伪造高密度平衡输入，因而不能声称已完成该项。若本轮不新增高密度实验，应明确在答辩中说明范围，并将摘要/结论里的90–180×限定为已测稀体系、硬件、精度设置和节点规模；不能笼统保留通用“两数量级”结论。

依据：[LAMMPS性能影响因素](https://docs.lammps.org/Speed_measure.html)、[PPPM参数](https://docs.lammps.org/kspace_modify.html)。
