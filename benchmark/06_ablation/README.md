# 06 当前RBMD的算法消融

使用同一已校准体系生成四组 MD，每组默认100步预热和2000步计时（预热不计入2000步）：

```bash
bash 06_ablation/prepare.sh --system systems/lj.json --output runs/lj-ablation-v1 --kmax 12 12 12 --repeats 5
```

组别为 Verlet+Ewald、Verlet+RBE、RBL+Ewald、RBL+RBE，另有 LAMMPS PPPM。kmax 必须先经01收敛；比较的是完整物理模型，不能去掉长程或短程力后称作消融。分项开销另加 `--isolated` 生成独立实验，分析同05。

RBMD 1.0与2.0版本对比已按用户决定取消，不安排单卡版本测试或跨平台移植，也不列为待完成项。旧入口 `run_rbmd1.sh` 已停用，执行只会提示取消，不启动程序。
