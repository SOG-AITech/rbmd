# 02 全盒采样与复制表示

最新结果见 [RESULT/02](../RESULT/02/RESULT.md)。

1. 准备相同副本的表示对照：

```bash
python3 02_full_box/materialize.py --system systems/water.base.json --replicate 3 3 3 --output runs/water-representation-v1
```

2. 先提交 `materialize` 生成 LAMMPS `expanded.data` 与参考力；再提交 `replicated` 和 `explicit`。两种表示要核对盒子、原子 ID/坐标、自动 α，并分别用 `common/analyze_force_samples.py` 对同一参考算 RMS。
3. **另外平衡整个 expanded.data 使副本去相关**，以新快照执行 `02_full_box/prepare.sh --system systems/water.base.json --data FULL_BOX_EQUILIBRATED.data --output NEW_RUN --cutoffs 9 --P 200 400 800 --p 100 --kmax 12 12 12`，操作与 01 相同。

相同副本的表示对照只能检查复制实现；旧的受限倒格子在完全相同副本上也可能通过，必须有去相关后的全盒精度实验。不要把副本加随机扰动后直接称为平衡体系。

2703原子基础输入已找到：先运行08的 `prepare_water_base.py` 补齐归档SPC/E系数，然后用 `water.base.json` 做3×3×3，得到72981原子。归档72981原子h2o.data是另一个输入，不能再复制后仍称72981。`materialize`使用CPU双精度并只计算run 0参考力；它生成的相同副本仍需全盒平衡。可用08的副本相关性工具检查坐标重复，非零差异不自动证明平衡。

源代码回归测试可在独立分配中运行 `$RBMD_SOURCE/tests/run_rbe_full_box.py`，但其小体系 smoke 不替代本实验的物理精度证据。
