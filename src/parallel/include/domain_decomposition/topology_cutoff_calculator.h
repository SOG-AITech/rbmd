#pragma once

#include <algorithm>

struct TopologyCutoffs {
  double bond_cutoff;  // 键相互作用所需截断（1.5 * max_bond + skin）
  double angle_cutoff; // 角相互作用所需截断（2.0 * max_bond + skin）
  double dihedral_cutoff; // 二面角相互作用所需截断（3.0 * max_bond + skin）
                          // 离平面角也适用
  double max_cutoff; // 最大截断（用于ghost通信）= MAX(bond, angle, dihedral)
};

class TopologyCutoffCalculator {
public:
  /**
   * @brief 从系统数据自动计算拓扑所需的截断距离
   *
   * 该方法会自动从 DataManager 获取：
   * - 拓扑信息（是否有bonds/angles/dihedrals/impropers）
   * - 最大平衡键长（从 Bond Coeffs 中计算得到）
   *
   * @param neighbor_skin 邻居列表skin
   * @return double 返回最大截断距离（用于ghost通信）
   */
  static double Calculate(double neighbor_skin = 0.0);

private:
  /**
   * @brief 从 DataManager 获取最大平衡键长
   * @return 最大平衡键长，如果未找到则返回默认值 1.8
   */
  static double GetMaxEquilibriumBondLength();
};