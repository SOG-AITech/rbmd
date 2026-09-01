#pragma once

#include <memory>
#include <tuple>
#include "common/types.h"
#include "data_manager/include/model/device_data.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace reader::mpi {

/**
 * @brief SpecialBond 构建器（CPU 侧构建 + 同步到 GPU）
 *
 * 责任边界（SRP）：
 * - 负责：基于“本地可见”的拓扑输入（bond/angle/dihedral）构建 1-2/1-3/1-4 special 邻接，
 *   并将结果转换为 GPU 可用的数据布局写入 `DeviceData`（例如 CSR/offset+ids+weights）。
 * - 不负责：MPI 通信本身、原子迁移/halo 交换、拓扑数据在消息中的打包协议。
 *
 * special/nspecial 约定：
 * - special 列表存储 GID/tag（按 1-2/1-3/1-4 顺序拼接）
 * - nspecial 为累积计数（长度 3，对应 1-2/1-3/1-4）
 * - 相关声明位置：
 *   - Host: `FullStructureData::_h_special_weights/_h_special_ids/_h_special_offsets/_h_special_offset_count`
 *   - Device: `DeviceData::_d_special_weights/_d_special_ids/_d_special_offsets/_d_special_count`
 * - 当前实现仍沿用 `_d_special_*` 这套布局，`DeviceData::d_nspecial/d_special` 为并行方案预留，尚未接入。
 *
 * 拓扑与通信语义：
 * - Topology-carry：bonds/angles/dihedrals 以 per-atom 列表随迁移携带
 * - Rendezvous：跨 rank pair 通过按需通信补齐/确认，路由为 atom_id % num_ranks（非 owner）
 *
 * 在架构中的典型调用链：
 * - 初始化阶段：Reader/MPI 分发完成后，收集 `local_atom_ids/ghost_atom_ids/local_bonds/...`，
 *   调用 `Build()` 生成 special 数据并同步到 GPU，供后续 force kernel 使用。
 * - 迁移阶段：DomainDecomposition 的 leaving 迁移会把“原子 + 拓扑”迁入新 rank，
 *   上层在迁移完成后选择合适时机触发 `Build()` 重建，保证 special 与当前拓扑一致。
 *
 * 使用示例：
 * @code
 * reader::mpi::SpecialBondBuilder builder;
 * builder.Build(local_ids, ghost_ids, bonds, angles, dihedrals, my_rank, num_ranks);
 * @endcode
 */
class SpecialBondBuilder {
public:
  using AdjacencyMap = std::unordered_map<rbmd::Id, std::vector<rbmd::Id> >;

  struct AdjacencyResult {
    AdjacencyMap adj12;
    AdjacencyMap adj13;
    AdjacencyMap adj14;
  };

  SpecialBondBuilder();

  // 测试友元类声明
  friend class SpecialBondBuilderTest;

  /// 构建special bond 邻接表
  /// @param local_atom_ids  本地native原子的global id
  /// @param ghost_atom_ids  ghost原子的 global id
  /// @param local_bonds  本地bonds（满足“1-2 本地可得性”；可由 per-atom topology 派生）
  /// @param local_angles 本地 angles（对齐 newton off：三端可见）
  /// @param local_dihedrals 本地 dihedrals（对齐 newton off：四端可见）
  /// @param my_rank 当前mpi rank
  /// @param num_ranks 总共的mpi rank
  void Build(std::vector<rbmd::Id> local_atom_ids,
             std::vector<rbmd::Id> ghost_atom_ids,
             std::vector<std::tuple<rbmd::Id, rbmd::Id> > local_bonds,
             const std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id> >&
             local_angles,
             const std::vector<std::tuple<
               int, rbmd::Id, rbmd::Id, rbmd::Id, rbmd::Id> >& local_dihedrals,
             int my_rank, int num_ranks);

  // 调试/测试入口：返回 trim 后的 1-2/1-3/1-4 邻接表（Global ID 空间）。
  AdjacencyResult BuildAdjacencyMaps(
      std::vector<rbmd::Id> local_atom_ids,
      std::vector<rbmd::Id> ghost_atom_ids,
      std::vector<std::tuple<rbmd::Id, rbmd::Id>> local_bonds,
      const std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id>>&
          local_angles,
      const std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id, rbmd::Id>>&
          local_dihedrals,
      int my_rank,
      int num_ranks);

private:
  std::shared_ptr<DeviceData> _device_data;
  std::vector<rbmd::Real> weights_;

  /// 仅构建本地原子的 1-2 邻接（Global ID 空间）
  ///
  /// 对齐 LAMMPS `onetwo_build_newton_off()` 语义：
  /// - 输入 `local_bonds` 已双端写入（由 ConvertGlobalToPerAtomTopology 保证）
  /// - 每条记录 (atom_i, atom_j) 表示 atom_i 的 bond partner 是 atom_j
  /// - 只需单向添加 adj12[atom_i].push_back(atom_j)
  /// - 只为 `local_atom_ids` 中的原子生成邻接（不为 ghost 生成）
  /// - 保序去重（保持插入顺序，避免 std::set 排序影响）
  ///
  /// @param local_atom_ids 本地 native 原子的 Global ID
  /// @param local_bonds 本地 bonds（已双端写入，满足"1-2 本地可得性"）
  /// @return 返回本地原子 1-2 邻接（去重后，Global ID 空间）
  AdjacencyMap BuildAdjacency12(
      const std::vector<rbmd::Id>& local_atom_ids,
      const std::vector<std::tuple<rbmd::Id, rbmd::Id>>& local_bonds);

  /// 构建本地原子的 1-3 邻接（Global ID 空间）
  ///
  /// 对齐 LAMMPS `onethree_build()`（ref/lammps/src/special.cpp:345-440）核心语义：
  /// - 以本地原子 c 作为"中心原子"，其 1-2 列表 N12(c) 已本地可得（newton_bond=off）
  /// - 对每个 u ∈ N12(c)，把 N12(c) 中除 u 外的其它邻居 w 作为 u 的 1-3 候选
  ///   - 若 u 本地可写：直接写入 adj13[u]
  ///   - 若 u 非本地：通过 RendezvousPairs(BUILD13, {u,w}) 路由给 u 的 owner 写入
  /// - adj13[u] 最终保序去重，并排除自身
  ///
  /// @param local_atom_ids 本地 native 原子的 Global ID
  /// @param adj12 1-2 邻接表（仅包含本地原子；来自 BuildAdjacency12 的输出）
  /// @param my_rank 当前 MPI rank
  /// @param num_ranks 总 MPI rank 数
  /// @return 返回本地原子 1-3 邻接（去重后，Global ID 空间）
  AdjacencyMap BuildAdjacency13(
      const std::vector<rbmd::Id>& local_atom_ids,
      const AdjacencyMap& adj12,
      int my_rank,
      int num_ranks);

  /// 构建本地原子的 1-4 邻接（Global ID 空间）
  ///
  /// 对齐 LAMMPS `onefour_build()`（ref/lammps/src/special.cpp:447-540）核心语义：
  /// - 以本地原子 c 作为"中心原子"，其 1-2 列表 N12(c) 和 1-3 列表 N13(c) 已可得
  /// - 对每个 w ∈ N13(c)，把 N12(c) 中的每个邻居 v 作为 w 的 1-4 候选
  ///   - 路径为 w -- ? -- c -- v，所以 w 和 v 是 1-4 邻居
  ///   - 若 w 本地可写：直接写入 adj14[w]
  ///   - 若 w 非本地：通过 RendezvousPairs(BUILD14, {w,v}) 路由给 w 的 owner 写入
  /// - adj14[w] 最终保序去重，并排除自身、1-2、1-3 邻居
  ///
  /// @param local_atom_ids 本地 native 原子的 Global ID
  /// @param adj12 1-2 邻接表（仅包含本地原子）
  /// @param adj13 1-3 邻接表（仅包含本地原子）
  /// @param my_rank 当前 MPI rank
  /// @param num_ranks 总 MPI rank 数
  /// @return 返回本地原子 1-4 邻接（去重后，Global ID 空间）
  AdjacencyMap BuildAdjacency14(
      const std::vector<rbmd::Id>& local_atom_ids,
      const AdjacencyMap& adj12,
      const AdjacencyMap& adj13,
      int my_rank,
      int num_ranks);

  // Rendezvous（pair-level）：用于跨 rank 按需补齐/确认 pair
  // - BUILD13/BUILD14：用于构建 1-3/1-4 候选
  // - ANGLE_TRIM/DIHEDRAL_TRIM：用于 trim 阶段把“应保留的 pair”送达正确端点 owner
  enum class RvousOp:int{
    BUILD13 = 0,
    BUILD14 = 1,
    ANGLE_TRIM = 2,
    DIHEDRAL_TRIM = 3,
  };

  struct PairRvous {
    rbmd::Id atom_id; // rendezvous 路由 key
    rbmd::Id partner_id; // 与 atom_id 相关的另一个原子
  };

  /// Pair-level rendezvous 通信：按需补齐/确认跨 rank 的 special pair
  ///
  /// 语义：
  /// - 输入 in_pairs 表示需要查询/确认的 pair（atom_id 为路由 key）
  /// - 路由规则：route_rank = atom_id % num_ranks（非 owner）
  /// - 通过 MPI_Alltoallv 交换 pair buffer 后，返回“应写入/保留”的 pair 列表
  ///
  /// op 说明：
  /// - BUILD13/BUILD14：用于生成 1-3/1-4 候选 pair
  /// - ANGLE_TRIM/DIHEDRAL_TRIM：用于 trim 阶段确认应保留的 pair
  ///
  /// 返回值：
  /// - 返回需要补齐/确认的 pair（由本地 owned 信息 + rendezvous 输出共同决定）
  /// - 调用方需在本地去重，并确保只写入本地端点（tid1 ∈ [0, N_local)）
  std::vector<PairRvous> RendezvousPairs(
    RvousOp op,
    std::vector<PairRvous>& in_pairs,
    int my_rank,
    int num_ranks
    );

  void ConvertAndSyncToDevice(
    const std::vector<rbmd::Id>& local_atom_ids,
    const std::vector<rbmd::Id>& ghost_atom_ids,
    const AdjacencyMap& adj12,
    const AdjacencyMap& adj13,
    const AdjacencyMap& adj14,
    int my_rank,
    int num_ranks
    );

  /// 计算 special bond 数据的 checksum（用于一致性验证）
  /// @param h_nspecial nspecial 数组 [nlocal * 3]
  /// @param h_special special 数组 [nlocal * maxspecial]
  /// @param nlocal 本地原子数
  /// @param maxspecial 最大 special 数
  /// @return checksum 值（64位无符号整数）
  uint64_t ComputeSpecialChecksum(
    const int* h_nspecial,
    const rbmd::Id* h_special,
    std::size_t nlocal,
    int maxspecial
    ) const;

  /// 路由：将 atom_gid 映射到 rendezvous 路由 rank（非 owner）
  /// @param atom_gid 全局唯一id
  /// @param num_ranks 总共的rank
  /// @return rendezvous 路由 rank（0 <= rank < num_ranks）
  /// “路由分解”不是找 owner，而是给每个 atom_id 选一个固定的中转/汇聚 rank。所有 rank 都用同一规则（atom_id %
  /// num_ranks）计算目标，所以同一个 atom_id 的请求一定到同一个地方，从而完成 rendezvous 的“按需补齐/确认 pair”
  int GetRendezvousRank(rbmd::Id atom_gid, int num_ranks);
};
} // namespace reader::mpi
