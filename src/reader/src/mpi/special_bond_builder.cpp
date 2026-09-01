#include "mpi/special_bond_builder.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#ifdef READER_ENABLE_MPI
#include "common/rbmd_define.h"  // for MPI_RBMD_ID
#include "mpi.h"
#endif

#include "data_manager.h"
#include "data_manager/include/structure_data/full_structure_data.h"

reader::mpi::SpecialBondBuilder::SpecialBondBuilder() {
  _device_data = DataManager::getInstance().getDeviceData();

  auto& md_data = DataManager::getInstance().getMDData();
  if (!md_data || !md_data->_structure_data) {
    return;
  }

  auto full =
      std::dynamic_pointer_cast<FullStructureData>(md_data->_structure_data);
  if (!full || !full->_h_special_weights || full->_num_special_weights == 0) {
    return;
  }

  weights_.assign(full->_h_special_weights,
                  full->_h_special_weights + full->_num_special_weights);
}

int reader::mpi::SpecialBondBuilder::GetRendezvousRank(rbmd::Id atom_gid,
                                                       int num_ranks) {
  if (num_ranks <= 0) {
    return 0;
  }
  const int mod = static_cast<int>(atom_gid % num_ranks);
  return mod < 0 ? (mod + num_ranks) : mod;
}

std::vector<reader::mpi::SpecialBondBuilder::PairRvous>
reader::mpi::SpecialBondBuilder::RendezvousPairs(
    RvousOp op, std::vector<PairRvous>& in_pairs, int my_rank, int num_ranks) {
  (void)my_rank;

#ifndef READER_ENABLE_MPI
  (void)num_ranks;
  (void)in_pairs;
  (void)op;
  return {};
#else
  // num_ranks==1：无需通信，仅做保序去重
  if (num_ranks <= 1) {
    std::vector<PairRvous> deduped;
    deduped.reserve(in_pairs.size());
    std::unordered_map<rbmd::Id, std::unordered_set<rbmd::Id>> seen;
    for (const auto& p : in_pairs) {
      auto& set = seen[p.atom_id];
      if (set.insert(p.partner_id).second) {
        deduped.push_back(p);
      }
    }
    return deduped;
  }

  // 注意：即使 in_pairs 为空，也必须参与 collective 通信（MPI_Alltoallv 是
  // collective） 否则其他 rank 发送给本 rank
  // 的数据无法被接收，导致通信不完整或死锁 对齐 LAMMPS comm->rendezvous()：即使
  // nsend=0 也要调用

  (void)op;

  auto build_displs = [](const std::vector<int>& counts,
                         std::vector<int>& displs, int& total) {
    displs.assign(counts.size(), 0);
    total = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      displs[i] = total;
      total += counts[i];
    }
  };

  // ============================
  // Stage 0: 构建 (atom_id -> owner_rank) 的 rendezvous 路由表
  // 说明：owner 信息来自各 rank 的本地原子列表，通过 atom_id%num_ranks 汇聚到
  // route rank
  // ============================
  std::vector<rbmd::Id> local_owned_ids;
  {
    auto& md_data = DataManager::getInstance().getMDData();
    const auto structure_info =
        md_data ? md_data->_structure_info_data : nullptr;
    const auto structure_data = md_data ? md_data->_structure_data : nullptr;

    const rbmd::Id local_atoms = (structure_info && structure_info->_num_atoms)
                                     ? *(structure_info->_num_atoms)
                                     : static_cast<rbmd::Id>(0);

    if (structure_data && structure_data->_h_atoms_id && local_atoms > 0) {
      const int nlocal = static_cast<int>(local_atoms);
      local_owned_ids.reserve(static_cast<std::size_t>(nlocal));
      for (int i = 0; i < nlocal; ++i) {
        local_owned_ids.push_back(structure_data->_h_atoms_id[i]);
      }
    }
  }

  // 发送 (atom_id, owner_rank)
  std::vector<int> owner_send_counts(num_ranks, 0);
  for (const rbmd::Id gid : local_owned_ids) {
    const int dst = GetRendezvousRank(gid, num_ranks);
    ++owner_send_counts[dst];
  }
  std::vector<int> owner_recv_counts(num_ranks, 0);
  MPI_Alltoall(owner_send_counts.data(), 1, MPI_INT, owner_recv_counts.data(),
               1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> owner_send_displs;
  std::vector<int> owner_recv_displs;
  int total_owner_send = 0;
  int total_owner_recv = 0;
  build_displs(owner_send_counts, owner_send_displs, total_owner_send);
  build_displs(owner_recv_counts, owner_recv_displs, total_owner_recv);

  std::vector<rbmd::Id> owner_send_flat(
      static_cast<std::size_t>(total_owner_send) * 2);
  {
    std::vector<int> cursor(num_ranks, 0);
    for (const rbmd::Id gid : local_owned_ids) {
      const int dst = GetRendezvousRank(gid, num_ranks);
      const int slot = owner_send_displs[dst] + cursor[dst]++;
      const std::size_t base = static_cast<std::size_t>(slot) * 2;
      owner_send_flat[base + 0] = gid;
      owner_send_flat[base + 1] = static_cast<rbmd::Id>(my_rank);
    }
  }

  std::vector<int> owner_send_counts_id(num_ranks, 0);
  std::vector<int> owner_send_displs_id(num_ranks, 0);
  std::vector<int> owner_recv_counts_id(num_ranks, 0);
  std::vector<int> owner_recv_displs_id(num_ranks, 0);
  for (int r = 0; r < num_ranks; ++r) {
    owner_send_counts_id[r] = owner_send_counts[r] * 2;
    owner_send_displs_id[r] = owner_send_displs[r] * 2;
    owner_recv_counts_id[r] = owner_recv_counts[r] * 2;
    owner_recv_displs_id[r] = owner_recv_displs[r] * 2;
  }

  std::vector<rbmd::Id> owner_recv_flat(
      static_cast<std::size_t>(total_owner_recv) * 2);
  MPI_Alltoallv(owner_send_flat.empty() ? nullptr : owner_send_flat.data(),
                owner_send_counts_id.data(), owner_send_displs_id.data(),
                MPI_RBMD_ID,
                owner_recv_flat.empty() ? nullptr : owner_recv_flat.data(),
                owner_recv_counts_id.data(), owner_recv_displs_id.data(),
                MPI_RBMD_ID, MPI_COMM_WORLD);

  std::unordered_map<rbmd::Id, int> owner_map;
  owner_map.reserve(static_cast<std::size_t>(total_owner_recv));
  for (int i = 0; i < total_owner_recv; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2;
    const rbmd::Id gid = owner_recv_flat[base + 0];
    const int owner = static_cast<int>(owner_recv_flat[base + 1]);
    owner_map[gid] = owner;
  }

  // ============================
  // Stage 1: 把 pair 按 route_rank=atom_id%num_ranks 汇聚
  // ============================
  std::vector<int> send_counts(num_ranks, 0);
  for (const auto& p : in_pairs) {
    const int dst = GetRendezvousRank(p.atom_id, num_ranks);
    ++send_counts[dst];
  }

  std::vector<int> recv_counts(num_ranks, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);

  std::vector<int> send_displs;
  std::vector<int> recv_displs;
  int total_send_pairs = 0;
  int total_recv_pairs = 0;
  build_displs(send_counts, send_displs, total_send_pairs);
  build_displs(recv_counts, recv_displs, total_recv_pairs);
  assert(total_send_pairs == static_cast<int>(in_pairs.size()));

  std::vector<rbmd::Id> send_flat(static_cast<std::size_t>(total_send_pairs) *
                                  2);
  {
    std::vector<int> cursor(num_ranks, 0);
    for (const auto& p : in_pairs) {
      const int dst = GetRendezvousRank(p.atom_id, num_ranks);
      const int slot = send_displs[dst] + cursor[dst]++;
      const std::size_t base = static_cast<std::size_t>(slot) * 2;
      send_flat[base + 0] = p.atom_id;
      send_flat[base + 1] = p.partner_id;
    }
  }

  std::vector<int> send_counts_id(num_ranks, 0);
  std::vector<int> send_displs_id(num_ranks, 0);
  std::vector<int> recv_counts_id(num_ranks, 0);
  std::vector<int> recv_displs_id(num_ranks, 0);
  for (int r = 0; r < num_ranks; ++r) {
    send_counts_id[r] = send_counts[r] * 2;
    send_displs_id[r] = send_displs[r] * 2;
    recv_counts_id[r] = recv_counts[r] * 2;
    recv_displs_id[r] = recv_displs[r] * 2;
  }

  std::vector<rbmd::Id> recv_flat(static_cast<std::size_t>(total_recv_pairs) *
                                  2);
  MPI_Alltoallv(send_flat.empty() ? nullptr : send_flat.data(),
                send_counts_id.data(), send_displs_id.data(), MPI_RBMD_ID,
                recv_flat.empty() ? nullptr : recv_flat.data(),
                recv_counts_id.data(), recv_displs_id.data(), MPI_RBMD_ID,
                MPI_COMM_WORLD);

  // ============================
  // Stage 2: 在 route rank 根据 owner_map 把 pair 转发给 atom_id 的 owner
  // ============================
  std::vector<std::vector<PairRvous>> out_by_owner(
      static_cast<std::size_t>(num_ranks));

  for (int i = 0; i < total_recv_pairs; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2;
    const rbmd::Id atom_id = recv_flat[base + 0];
    const rbmd::Id partner_id = recv_flat[base + 1];

    auto it = owner_map.find(atom_id);
    if (it == owner_map.end()) {
      continue;
    }
    const int owner = it->second;
    if (owner < 0 || owner >= num_ranks) {
      continue;
    }
    out_by_owner[static_cast<std::size_t>(owner)].push_back(
        PairRvous{atom_id, partner_id});
  }

  std::vector<int> resp_send_counts(num_ranks, 0);
  for (int r = 0; r < num_ranks; ++r) {
    resp_send_counts[r] =
        static_cast<int>(out_by_owner[static_cast<std::size_t>(r)].size());
  }

  std::vector<int> resp_recv_counts(num_ranks, 0);
  MPI_Alltoall(resp_send_counts.data(), 1, MPI_INT, resp_recv_counts.data(), 1,
               MPI_INT, MPI_COMM_WORLD);

  std::vector<int> resp_send_displs;
  std::vector<int> resp_recv_displs;
  int total_resp_send_pairs = 0;
  int total_resp_recv_pairs = 0;
  build_displs(resp_send_counts, resp_send_displs, total_resp_send_pairs);
  build_displs(resp_recv_counts, resp_recv_displs, total_resp_recv_pairs);

  std::vector<rbmd::Id> resp_send_flat(
      static_cast<std::size_t>(total_resp_send_pairs) * 2);
  for (int dst = 0; dst < num_ranks; ++dst) {
    const auto& out = out_by_owner[static_cast<std::size_t>(dst)];
    const int offset = resp_send_displs[dst];
    for (std::size_t i = 0; i < out.size(); ++i) {
      const int slot = offset + static_cast<int>(i);
      const std::size_t base = static_cast<std::size_t>(slot) * 2;
      resp_send_flat[base + 0] = out[i].atom_id;
      resp_send_flat[base + 1] = out[i].partner_id;
    }
  }

  std::vector<rbmd::Id> resp_recv_flat(
      static_cast<std::size_t>(total_resp_recv_pairs) * 2);
  std::vector<int> resp_send_counts_id(num_ranks, 0);
  std::vector<int> resp_send_displs_id(num_ranks, 0);
  std::vector<int> resp_recv_counts_id(num_ranks, 0);
  std::vector<int> resp_recv_displs_id(num_ranks, 0);
  for (int r = 0; r < num_ranks; ++r) {
    resp_send_counts_id[r] = resp_send_counts[r] * 2;
    resp_send_displs_id[r] = resp_send_displs[r] * 2;
    resp_recv_counts_id[r] = resp_recv_counts[r] * 2;
    resp_recv_displs_id[r] = resp_recv_displs[r] * 2;
  }

  MPI_Alltoallv(resp_send_flat.empty() ? nullptr : resp_send_flat.data(),
                resp_send_counts_id.data(), resp_send_displs_id.data(),
                MPI_RBMD_ID,
                resp_recv_flat.empty() ? nullptr : resp_recv_flat.data(),
                resp_recv_counts_id.data(), resp_recv_displs_id.data(),
                MPI_RBMD_ID, MPI_COMM_WORLD);

  // ============================
  // T015: 响应解析 + 去重
  // ============================
  // 规则：同一个 atom_id 下对 partner_id 去重（保序：保留第一次出现的顺序）。
  std::vector<PairRvous> deduped;
  deduped.reserve(static_cast<std::size_t>(total_resp_recv_pairs));

  std::unordered_map<rbmd::Id, std::unordered_set<rbmd::Id>> seen;
  seen.reserve(static_cast<std::size_t>(total_resp_recv_pairs));

  for (int i = 0; i < total_resp_recv_pairs; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2;
    const rbmd::Id atom_id = resp_recv_flat[base + 0];
    const rbmd::Id partner_id = resp_recv_flat[base + 1];

    auto& set = seen[atom_id];
    if (set.insert(partner_id).second) {
      deduped.push_back(PairRvous{atom_id, partner_id});
    }
  }

  return deduped;
#endif
}

reader::mpi::SpecialBondBuilder::AdjacencyMap
reader::mpi::SpecialBondBuilder::BuildAdjacency12(
    const std::vector<rbmd::Id>& local_atom_ids,
    const std::vector<std::tuple<rbmd::Id, rbmd::Id>>& local_bonds) {
  // 构建本地原子 ID 集合，用于快速判断是否为本地原子
  std::unordered_set<rbmd::Id> local_atom_set(local_atom_ids.begin(),
                                              local_atom_ids.end());

  AdjacencyMap adj12;

  // 遍历所有 bond，单向添加邻接
  // 对齐 LAMMPS onetwo_build_newton_off()：
  // - newton_bond=off 时，local_bonds 已双端写入（由
  // ConvertGlobalToPerAtomTopology 保证）
  // - 每条记录 (atom_i, atom_j) 表示 atom_i 的 bond partner 是 atom_j
  // - 只需单向添加 adj12[atom_i].push_back(atom_j)
  for (const auto& bond : local_bonds) {
    const rbmd::Id atom_i = std::get<0>(bond);
    const rbmd::Id atom_j = std::get<1>(bond);

    // 只为本地原子生成邻接（不为 ghost 生成）
    if (local_atom_set.count(atom_i) > 0) {
      adj12[atom_i].push_back(atom_j);
    }
  }

  // 保序去重：保持插入顺序，避免 std::set 排序影响
  // 对齐 LAMMPS dedup() 的语义
  for (auto& [atom_id, neighbors] : adj12) {
    std::unordered_set<rbmd::Id> seen;
    std::vector<rbmd::Id> unique_neighbors;
    unique_neighbors.reserve(neighbors.size());

    for (const rbmd::Id neighbor : neighbors) {
      if (seen.insert(neighbor).second) {
        unique_neighbors.push_back(neighbor);
      }
    }

    neighbors = std::move(unique_neighbors);
  }

  return adj12;
}

reader::mpi::SpecialBondBuilder::AdjacencyMap
reader::mpi::SpecialBondBuilder::BuildAdjacency13(
    const std::vector<rbmd::Id>& local_atom_ids, const AdjacencyMap& adj12,
    int my_rank, int num_ranks) {
  // 对齐 LAMMPS onethree_build()（ref/lammps/src/special.cpp:345-440）
  // newton_bond=off：中心原子 c 的 1-2 列表 N12(c) 本地可得。
  // 由中心原子 c 产生其每个 1-2 邻居 u 的 1-3 候选：N12(c) \ {u}

  std::unordered_set<rbmd::Id> local_set(local_atom_ids.begin(),
                                         local_atom_ids.end());

  AdjacencyMap adj13;
  std::vector<PairRvous> pending;

  for (const rbmd::Id center : local_atom_ids) {
    auto it = adj12.find(center);
    if (it == adj12.end()) {
      continue;
    }

    const auto& neighbors = it->second;
    const std::size_t deg = neighbors.size();
    if (deg <= 1) {
      continue;
    }

    for (std::size_t j = 0; j < deg; ++j) {
      const rbmd::Id u = neighbors[j];
      for (std::size_t k = 0; k < deg; ++k) {
        if (k == j) {
          continue;
        }
        const rbmd::Id w = neighbors[k];

        if (local_set.count(u) > 0) {
          adj13[u].push_back(w);
        } else {
          pending.push_back(PairRvous{u, w});
        }
      }
    }
  }

  // 通过 rendezvous 路由把 (u,w) 送到 u 的 owner
  auto routed = RendezvousPairs(RvousOp::BUILD13, pending, my_rank, num_ranks);
  for (const auto& p : routed) {
    if (local_set.count(p.atom_id) > 0) {
      adj13[p.atom_id].push_back(p.partner_id);
    }
  }

  // 保序去重 + 排除自身
  for (auto it = adj13.begin(); it != adj13.end();) {
    const rbmd::Id atom_id = it->first;
    auto& neighbors = it->second;

    std::unordered_set<rbmd::Id> seen;
    seen.insert(atom_id);
    std::vector<rbmd::Id> unique_neighbors;
    unique_neighbors.reserve(neighbors.size());

    for (const rbmd::Id neighbor : neighbors) {
      if (seen.insert(neighbor).second) {
        unique_neighbors.push_back(neighbor);
      }
    }

    neighbors = std::move(unique_neighbors);
    if (neighbors.empty()) {
      it = adj13.erase(it);
    } else {
      ++it;
    }
  }

  return adj13;
}

reader::mpi::SpecialBondBuilder::AdjacencyMap
reader::mpi::SpecialBondBuilder::BuildAdjacency14(
    const std::vector<rbmd::Id>& local_atom_ids, const AdjacencyMap& adj12,
    const AdjacencyMap& adj13, int my_rank, int num_ranks) {
  // 对齐 LAMMPS onefour_build()（ref/lammps/src/special.cpp:447-540）
  // newton_bond=off：中心原子 c 的 N12(c) 和 N13(c) 本地可得。
  // 对每个 w ∈ N13(c)，把 N12(c) 中的 v 作为 w 的 1-4 候选
  // 路径：w -- ? -- c -- v，所以 w 和 v 是 1-4 邻居

  std::unordered_set<rbmd::Id> local_set(local_atom_ids.begin(),
                                         local_atom_ids.end());

  AdjacencyMap adj14;
  std::vector<PairRvous> pending;

  for (const rbmd::Id center : local_atom_ids) {
    // 获取 center 的 1-3 邻居
    auto it13 = adj13.find(center);
    if (it13 == adj13.end()) {
      continue;
    }
    const auto& neighbors_13 = it13->second;

    // 获取 center 的 1-2 邻居
    auto it12 = adj12.find(center);
    if (it12 == adj12.end()) {
      continue;
    }
    const auto& neighbors_12 = it12->second;

    // 对每个 1-3 邻居 w 和 1-2 邻居 v，(w, v) 是 1-4 关系
    for (const rbmd::Id w : neighbors_13) {
      for (const rbmd::Id v : neighbors_12) {
        if (local_set.count(w) > 0) {
          adj14[w].push_back(v);
        } else {
          pending.push_back(PairRvous{w, v});
        }
      }
    }
  }

  // 通过 rendezvous 路由把 (w, v) 送到 w 的 owner
  auto routed = RendezvousPairs(RvousOp::BUILD14, pending, my_rank, num_ranks);
  for (const auto& p : routed) {
    if (local_set.count(p.atom_id) > 0) {
      adj14[p.atom_id].push_back(p.partner_id);
    }
  }

  // 保序去重 + 排除自身、1-2、1-3 邻居
  for (auto it = adj14.begin(); it != adj14.end();) {
    const rbmd::Id atom_id = it->first;
    auto& neighbors = it->second;

    // 构建排除集合：自身 + 1-2 + 1-3
    std::unordered_set<rbmd::Id> exclude;
    exclude.insert(atom_id);

    auto it12 = adj12.find(atom_id);
    if (it12 != adj12.end()) {
      for (const rbmd::Id nbr : it12->second) {
        exclude.insert(nbr);
      }
    }

    auto it13 = adj13.find(atom_id);
    if (it13 != adj13.end()) {
      for (const rbmd::Id nbr : it13->second) {
        exclude.insert(nbr);
      }
    }

    std::unordered_set<rbmd::Id> seen;
    std::vector<rbmd::Id> unique_neighbors;
    unique_neighbors.reserve(neighbors.size());

    for (const rbmd::Id neighbor : neighbors) {
      // 排除自身、1-2、1-3 邻居
      if (exclude.count(neighbor) > 0) {
        continue;
      }
      if (seen.insert(neighbor).second) {
        unique_neighbors.push_back(neighbor);
      }
    }

    neighbors = std::move(unique_neighbors);
    if (neighbors.empty()) {
      it = adj14.erase(it);
    } else {
      ++it;
    }
  }

  return adj14;
}

reader::mpi::SpecialBondBuilder::AdjacencyResult
reader::mpi::SpecialBondBuilder::BuildAdjacencyMaps(
    std::vector<rbmd::Id> local_atom_ids, std::vector<rbmd::Id> ghost_atom_ids,
    std::vector<std::tuple<rbmd::Id, rbmd::Id>> local_bonds,
    const std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id>>&
        local_angles,
    const std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id, rbmd::Id>>&
        local_dihedrals,
    int my_rank, int num_ranks) {
  (void)ghost_atom_ids;
  // ========== 阶段 1：构建 1-2 邻接 ==========
  // adj12: 只包含本地原子的 1-2 邻接（用于最终输出）
  AdjacencyMap adj12 = BuildAdjacency12(local_atom_ids, local_bonds);

  // 统计 1-2 邻接信息
  std::size_t total_degree = 0;
  std::size_t max_degree = 0;
  for (const auto& [atom_id, neighbors] : adj12) {
    total_degree += neighbors.size();
    max_degree = std::max(max_degree, neighbors.size());
  }

  const std::size_t num_atoms_with_bonds = adj12.size();
  const double avg_degree = num_atoms_with_bonds > 0
                                ? static_cast<double>(total_degree) /
                                      static_cast<double>(num_atoms_with_bonds)
                                : 0.0;

#ifdef READER_ENABLE_MPI
  // MPI 汇总统计
  std::size_t global_max_degree = 0;
  MPI_Allreduce(&max_degree, &global_max_degree, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::cout << "[SpecialBondBuilder] 1-2 adjacency built: "
              << "global_max_degree=" << global_max_degree << std::endl;
  }

  // 每个 rank 打印本地统计
  std::cout << "[SpecialBondBuilder] Rank " << my_rank
            << ": local_atoms=" << local_atom_ids.size()
            << ", atoms_with_bonds=" << num_atoms_with_bonds
            << ", avg_degree=" << avg_degree << ", max_degree=" << max_degree
            << std::endl;
#else
  std::cout << "[SpecialBondBuilder] 1-2 adjacency built: " << "local_atoms="
            << local_atom_ids.size()
            << ", atoms_with_bonds=" << num_atoms_with_bonds
            << ", avg_degree=" << avg_degree << ", max_degree=" << max_degree
            << std::endl;
#endif

  // ========== 阶段 2：1-3 推导（T016）==========
  // 对齐 LAMMPS onethree_build()（ref/lammps/src/special.cpp:345-440）
  // newton_bond=off：中心原子 c 的 N12(c) 本地可得；跨 rank 通过 rendezvous
  // 路由到端点 owner
  AdjacencyMap adj13 =
      BuildAdjacency13(local_atom_ids, adj12, my_rank, num_ranks);

  // angle_trim：只保留由 angle/dihedral 拓扑支持的 1-3（对齐 LAMMPS
  // Special::angle_trim） 注意：MPI collective 必须所有 rank
  // 同步进入，不能用“本 rank 是否有 angles/dihedrals”做分支。
  int global_has_angle_or_dihedral = 0;
#ifdef READER_ENABLE_MPI
  {
    const int local_has =
        (!local_angles.empty() || !local_dihedrals.empty()) ? 1 : 0;
    MPI_Allreduce(&local_has, &global_has_angle_or_dihedral, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
  }
#else
  global_has_angle_or_dihedral =
      (!local_angles.empty() || !local_dihedrals.empty()) ? 1 : 0;
#endif

  if (!global_has_angle_or_dihedral) {
    // 如果全局没有任何 angle/dihedral 定义，删除所有 1-3（对齐 LAMMPS）
    adj13.clear();
  } else {
    std::unordered_set<rbmd::Id> local_set(local_atom_ids.begin(),
                                           local_atom_ids.end());

    std::unordered_map<rbmd::Id, std::unordered_set<rbmd::Id>> keep;
    std::vector<PairRvous> pending_keep;

    auto add_keep = [&](rbmd::Id atom_id, rbmd::Id partner_id) {
      if (local_set.count(atom_id) > 0) {
        keep[atom_id].insert(partner_id);
      } else {
        pending_keep.push_back(PairRvous{atom_id, partner_id});
      }
    };

    // angle: keep (atom1,atom3) and (atom3,atom1)
    for (const auto& a : local_angles) {
      const rbmd::Id atom1 = std::get<1>(a);
      const rbmd::Id atom2 = std::get<2>(a);
      const rbmd::Id atom3 = std::get<3>(a);
      if (local_set.count(atom2) == 0) {
        continue;  // 只处理“我拥有 atom2”的记录
      }
      add_keep(atom1, atom3);
      add_keep(atom3, atom1);
    }

    // dihedral: keep (atom1,atom3) and (atom2,atom4) (both directions)
    for (const auto& d : local_dihedrals) {
      const rbmd::Id atom1 = std::get<1>(d);
      const rbmd::Id atom2 = std::get<2>(d);
      const rbmd::Id atom3 = std::get<3>(d);
      const rbmd::Id atom4 = std::get<4>(d);
      if (local_set.count(atom2) == 0) {
        continue;  // 只处理“我拥有 atom2”的记录
      }
      add_keep(atom1, atom3);
      add_keep(atom3, atom1);
      add_keep(atom2, atom4);
      add_keep(atom4, atom2);
    }

    // 关键：即使 pending_keep 为空，也必须参与 rendezvous（内部是
    // collective）。
    auto delivered =
        RendezvousPairs(RvousOp::ANGLE_TRIM, pending_keep, my_rank, num_ranks);
    for (const auto& p : delivered) {
      if (local_set.count(p.atom_id) > 0) {
        keep[p.atom_id].insert(p.partner_id);
      }
    }

    // filter adj13
    for (auto it = adj13.begin(); it != adj13.end();) {
      const rbmd::Id u = it->first;
      auto& nbrs = it->second;
      auto kit = keep.find(u);
      if (kit == keep.end()) {
        it = adj13.erase(it);
        continue;
      }

      const auto& kset = kit->second;
      std::vector<rbmd::Id> filtered;
      filtered.reserve(nbrs.size());
      for (const rbmd::Id w : nbrs) {
        if (kset.count(w) > 0) {
          filtered.push_back(w);
        }
      }
      nbrs = std::move(filtered);
      if (nbrs.empty()) {
        it = adj13.erase(it);
      } else {
        ++it;
      }
    }
  }

  // 统计 1-3 邻接信息
  std::size_t total_13_degree = 0;
  std::size_t max_13_degree = 0;
  for (const auto& [atom_id, neighbors] : adj13) {
    total_13_degree += neighbors.size();
    max_13_degree = std::max(max_13_degree, neighbors.size());
  }

  const std::size_t num_atoms_with_13 = adj13.size();
  const double avg_13_degree = num_atoms_with_13 > 0
                                   ? static_cast<double>(total_13_degree) /
                                         static_cast<double>(num_atoms_with_13)
                                   : 0.0;

#ifdef READER_ENABLE_MPI
  std::size_t global_max_13_degree = 0;
  MPI_Allreduce(&max_13_degree, &global_max_13_degree, 1, MPI_UNSIGNED_LONG,
                MPI_MAX, MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::cout << "[SpecialBondBuilder] 1-3 adjacency built: "
              << "global_max_degree=" << global_max_13_degree << std::endl;
  }

  std::cout << "[SpecialBondBuilder] Rank " << my_rank
            << ": atoms_with_1-3=" << num_atoms_with_13
            << ", avg_degree=" << avg_13_degree
            << ", max_degree=" << max_13_degree << std::endl;
#else
  std::cout << "[SpecialBondBuilder] 1-3 adjacency built: " << "atoms_with_1-3="
            << num_atoms_with_13 << ", avg_degree=" << avg_13_degree
            << ", max_degree=" << max_13_degree << std::endl;
#endif

  // ========== 阶段 3：1-4 推导（T017）==========
  // 对齐 LAMMPS onefour_build()（ref/lammps/src/special.cpp:447-540）
  // newton_bond=off：中心原子 c 的 N12(c) 和 N13(c) 本地可得；跨 rank 通过
  // rendezvous 路由到端点 owner
  AdjacencyMap adj14 =
      BuildAdjacency14(local_atom_ids, adj12, adj13, my_rank, num_ranks);

  // dihedral_trim：只保留由 dihedral 拓扑支持的 1-4（对齐 LAMMPS
  // Special::dihedral_trim） 同样需要全局一致的分支，避免某些 rank 进入
  // collective 而其他 rank 不进入。
  int global_has_dihedral = 0;
#ifdef READER_ENABLE_MPI
  {
    const int local_has = (!local_dihedrals.empty()) ? 1 : 0;
    MPI_Allreduce(&local_has, &global_has_dihedral, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
  }
#else
  global_has_dihedral = (!local_dihedrals.empty()) ? 1 : 0;
#endif

  if (!global_has_dihedral) {
    // 如果全局没有 dihedral 定义，删除所有 1-4（对齐 LAMMPS）
    adj14.clear();
  } else {
    std::unordered_set<rbmd::Id> local_set(local_atom_ids.begin(),
                                           local_atom_ids.end());

    std::unordered_map<rbmd::Id, std::unordered_set<rbmd::Id>> keep;
    std::vector<PairRvous> pending_keep;

    auto add_keep = [&](rbmd::Id atom_id, rbmd::Id partner_id) {
      if (local_set.count(atom_id) > 0) {
        keep[atom_id].insert(partner_id);
      } else {
        pending_keep.push_back(PairRvous{atom_id, partner_id});
      }
    };

    // dihedral: keep (atom1, atom4) and (atom4, atom1)
    // 对齐 LAMMPS dihedral_trim：只处理"我拥有 atom2"的记录以避免重复
    for (const auto& d : local_dihedrals) {
      const rbmd::Id atom1 = std::get<1>(d);
      const rbmd::Id atom2 = std::get<2>(d);
      const rbmd::Id atom4 = std::get<4>(d);
      if (local_set.count(atom2) == 0) {
        continue;  // 只处理"我拥有 atom2"的记录
      }
      add_keep(atom1, atom4);
      add_keep(atom4, atom1);
    }

    // 关键：即使 pending_keep 为空，也必须参与 rendezvous（内部是
    // collective）。
    auto delivered = RendezvousPairs(RvousOp::DIHEDRAL_TRIM, pending_keep,
                                     my_rank, num_ranks);
    for (const auto& p : delivered) {
      if (local_set.count(p.atom_id) > 0) {
        keep[p.atom_id].insert(p.partner_id);
      }
    }

    // filter adj14
    for (auto it = adj14.begin(); it != adj14.end();) {
      const rbmd::Id u = it->first;
      auto& nbrs = it->second;
      auto kit = keep.find(u);
      if (kit == keep.end()) {
        it = adj14.erase(it);
        continue;
      }

      const auto& kset = kit->second;
      std::vector<rbmd::Id> filtered;
      filtered.reserve(nbrs.size());
      for (const rbmd::Id x : nbrs) {
        if (kset.count(x) > 0) {
          filtered.push_back(x);
        }
      }
      nbrs = std::move(filtered);
      if (nbrs.empty()) {
        it = adj14.erase(it);
      } else {
        ++it;
      }
    }
  }

  // 统计 1-4 邻接信息
  std::size_t total_14_degree = 0;
  std::size_t max_14_degree = 0;
  for (const auto& [atom_id, neighbors] : adj14) {
    total_14_degree += neighbors.size();
    max_14_degree = std::max(max_14_degree, neighbors.size());
  }

  const std::size_t num_atoms_with_14 = adj14.size();
  const double avg_14_degree = num_atoms_with_14 > 0
                                   ? static_cast<double>(total_14_degree) /
                                         static_cast<double>(num_atoms_with_14)
                                   : 0.0;

#ifdef READER_ENABLE_MPI
  std::size_t global_max_14_degree = 0;
  MPI_Allreduce(&max_14_degree, &global_max_14_degree, 1, MPI_UNSIGNED_LONG,
                MPI_MAX, MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::cout << "[SpecialBondBuilder] 1-4 adjacency built: "
              << "global_max_degree=" << global_max_14_degree << std::endl;
  }

  std::cout << "[SpecialBondBuilder] Rank " << my_rank
            << ": atoms_with_1-4=" << num_atoms_with_14
            << ", avg_degree=" << avg_14_degree
            << ", max_degree=" << max_14_degree << std::endl;
#else
  std::cout << "[SpecialBondBuilder] 1-4 adjacency built: " << "atoms_with_1-4="
            << num_atoms_with_14 << ", avg_degree=" << avg_14_degree
            << ", max_degree=" << max_14_degree << std::endl;
#endif

  // ========== 阶段 5：转换并同步到 GPU ==========
  ConvertAndSyncToDevice(local_atom_ids, ghost_atom_ids, adj12, adj13, adj14,
                         my_rank, num_ranks);

  AdjacencyResult out;
  out.adj12 = std::move(adj12);
  out.adj13 = std::move(adj13);
  out.adj14 = std::move(adj14);
  return out;
}

void reader::mpi::SpecialBondBuilder::ConvertAndSyncToDevice(
    const std::vector<rbmd::Id>& local_atom_ids,
    const std::vector<rbmd::Id>& ghost_atom_ids, const AdjacencyMap& adj12,
    const AdjacencyMap& adj13, const AdjacencyMap& adj14, int my_rank,
    int num_ranks) {
  (void)ghost_atom_ids;

  const std::size_t nlocal = local_atom_ids.size();

  if (nlocal == 0) {
    return;
  }

  // ============================
  // 阶段 1: 合并 adj12/adj13/adj14 为 nspecial/special
  // 对齐 LAMMPS Special::combine()（ref/lammps/src/special.cpp:756-924）
  // ============================

  // Host 侧数组：nspecial[nlocal*3]
  std::vector<int> h_nspecial(nlocal * 3, 0);

  // 计算本地 maxspecial：对每个原子合并并去重 special 列表
  // 注意：为避免使用固定大小临时缓冲导致截断，这里采用“两遍构建”
  // - 第 1 遍：仅计算 nspecial 和 local_maxspecial
  // - 第 2 遍：在得到 global_maxspecial 后，重新生成 special 并写入 h_special
  int local_maxspecial = 0;

  auto build_special_for_atom = [&](rbmd::Id atom_gid,
                                   std::vector<rbmd::Id>* out_list,
                                   int& n12,
                                   int& n13,
                                   int& n14) {
    std::unordered_set<rbmd::Id> seen;
    seen.insert(atom_gid);  // 排除自身

    if (out_list) {
      out_list->clear();
    }

    // 1-2
    n12 = 0;
    auto it12 = adj12.find(atom_gid);
    if (it12 != adj12.end()) {
      for (const rbmd::Id partner : it12->second) {
        if (seen.insert(partner).second) {
          if (out_list) {
            out_list->push_back(partner);
          }
          ++n12;
        }
      }
    }

    // 1-3
    n13 = n12;
    auto it13 = adj13.find(atom_gid);
    if (it13 != adj13.end()) {
      for (const rbmd::Id partner : it13->second) {
        if (seen.insert(partner).second) {
          if (out_list) {
            out_list->push_back(partner);
          }
          ++n13;
        }
      }
    }

    // 1-4
    n14 = n13;
    auto it14 = adj14.find(atom_gid);
    if (it14 != adj14.end()) {
      for (const rbmd::Id partner : it14->second) {
        if (seen.insert(partner).second) {
          if (out_list) {
            out_list->push_back(partner);
          }
          ++n14;
        }
      }
    }
  };

  // 第 1 遍：只计算 nspecial + local_maxspecial
  for (std::size_t tid = 0; tid < nlocal; ++tid) {
    const rbmd::Id atom_gid = local_atom_ids[tid];

    int n12 = 0, n13 = 0, n14 = 0;
    build_special_for_atom(atom_gid, nullptr, n12, n13, n14);

    // 设置累积计数（对齐 LAMMPS nspecial 语义）
    h_nspecial[tid * 3 + 0] = n12;
    h_nspecial[tid * 3 + 1] = n13;
    h_nspecial[tid * 3 + 2] = n14;

    local_maxspecial = std::max(local_maxspecial, n14);
  }

  // ============================
  // 阶段 2: MPI_Allreduce 计算全局 maxspecial
  // ============================
  int global_maxspecial = local_maxspecial;
#ifdef READER_ENABLE_MPI
  if (num_ranks > 1) {
    MPI_Allreduce(&local_maxspecial, &global_maxspecial, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
  }
#else
  (void)num_ranks;
#endif

  if (global_maxspecial <= 0) {
    // 无 special bonds，清空 device 数据
    if (_device_data) {
      _device_data->maxspecial = 0;
      _device_data->d_nspecial.clear();
      _device_data->d_special.clear();
    }
#ifdef READER_ENABLE_MPI
    if (my_rank == 0) {
      std::cout << "[SpecialBondBuilder] No special bonds found, skipping "
                   "device sync."
                << std::endl;
    }
#else
    std::cout
        << "[SpecialBondBuilder] No special bonds found, skipping device sync."
        << std::endl;
#endif
    return;
  }

  // ============================
  // 阶段 3: 准备最终的 h_special（使用实际的 global_maxspecial）
  // ============================
  std::vector<rbmd::Id> h_special(nlocal * global_maxspecial, 0);

  // 第 2 遍：生成 special 列表并写入 h_special（不截断）
  std::vector<rbmd::Id> combined_list;
  for (std::size_t tid = 0; tid < nlocal; ++tid) {
    const rbmd::Id atom_gid = local_atom_ids[tid];

    int n12 = 0, n13 = 0, n14 = 0;
    build_special_for_atom(atom_gid, &combined_list, n12, n13, n14);

    // 防御性：确保不会写越界
    const std::size_t copy_count =
        std::min(static_cast<std::size_t>(n14),
                 static_cast<std::size_t>(global_maxspecial));

    for (std::size_t k = 0; k < copy_count; ++k) {
      h_special[tid * global_maxspecial + k] = combined_list[k];
    }
  }

  // ============================
  // 阶段 4: 契约校验与统计
  // ============================
  // 校验：nspecial 累积关系正确、不超过 maxspecial
  bool validation_ok = true;
  for (std::size_t tid = 0; tid < nlocal; ++tid) {
    const int n12 = h_nspecial[tid * 3 + 0];
    const int n13 = h_nspecial[tid * 3 + 1];
    const int n14 = h_nspecial[tid * 3 + 2];

    if (!(0 <= n12 && n12 <= n13 && n13 <= n14 && n14 <= global_maxspecial)) {
      std::cerr << "[SpecialBondBuilder] Error: atom tid=" << tid
                << " (gid=" << local_atom_ids[tid]
                << ") has invalid nspecial: [" << n12 << ", " << n13 << ", "
                << n14 << "], maxspecial=" << global_maxspecial << std::endl;
      validation_ok = false;
    }
  }

  if (!validation_ok) {
    std::cerr << "[SpecialBondBuilder] Validation failed on rank " << my_rank
              << ", aborting device sync." << std::endl;
    return;
  }

  // 统计：计算 nspecial[0/1/2] 分布（T021）
  std::size_t total_n12 = 0, total_n13 = 0, total_n14 = 0;
  int max_n12 = 0, max_n13 = 0, max_n14 = 0;
  int min_n12 = global_maxspecial, min_n13 = global_maxspecial,
      min_n14 = global_maxspecial;
  std::size_t atoms_with_special = 0;

  for (std::size_t tid = 0; tid < nlocal; ++tid) {
    const int n12 = h_nspecial[tid * 3 + 0];
    const int n13 = h_nspecial[tid * 3 + 1];
    const int n14 = h_nspecial[tid * 3 + 2];

    total_n12 += n12;
    total_n13 += n13;
    total_n14 += n14;

    max_n12 = std::max(max_n12, n12);
    max_n13 = std::max(max_n13, n13);
    max_n14 = std::max(max_n14, n14);

    if (n14 > 0) {
      min_n12 = std::min(min_n12, n12);
      min_n13 = std::min(min_n13, n13);
      min_n14 = std::min(min_n14, n14);
      ++atoms_with_special;
    }
  }

  // 如果没有原子有 special bonds，重置 min 值
  if (atoms_with_special == 0) {
    min_n12 = min_n13 = min_n14 = 0;
  }

  const double avg_n12 =
      nlocal > 0 ? static_cast<double>(total_n12) / static_cast<double>(nlocal)
                 : 0.0;
  const double avg_n13 =
      nlocal > 0 ? static_cast<double>(total_n13) / static_cast<double>(nlocal)
                 : 0.0;
  const double avg_n14 =
      nlocal > 0 ? static_cast<double>(total_n14) / static_cast<double>(nlocal)
                 : 0.0;

#ifdef READER_ENABLE_MPI
  if (my_rank == 0) {
    std::cout << "[SpecialBondBuilder] Global maxspecial = "
              << global_maxspecial << std::endl;
  }

  // T021: 详细的 nspecial 统计信息
  std::cout << "[SpecialBondBuilder] Rank " << my_rank << " statistics:\n"
            << "  nlocal=" << nlocal
            << ", atoms_with_special=" << atoms_with_special << "\n"
            << "  1-2: avg=" << avg_n12 << ", min=" << min_n12
            << ", max=" << max_n12 << "\n"
            << "  1-3: avg=" << avg_n13 << ", min=" << min_n13
            << ", max=" << max_n13 << "\n"
            << "  1-4: avg=" << avg_n14 << ", min=" << min_n14
            << ", max=" << max_n14 << std::endl;
#else
  std::cout << "[SpecialBondBuilder] Statistics:\n"
            << "  nlocal=" << nlocal
            << ", atoms_with_special=" << atoms_with_special
            << ", global_maxspecial=" << global_maxspecial << "\n"
            << "  1-2: avg=" << avg_n12 << ", min=" << min_n12
            << ", max=" << max_n12 << "\n"
            << "  1-3: avg=" << avg_n13 << ", min=" << min_n13
            << ", max=" << max_n13 << "\n"
            << "  1-4: avg=" << avg_n14 << ", min=" << min_n14
            << ", max=" << max_n14 << std::endl;
#endif

  // ============================
  // 阶段 5: 写入 FullStructureData（Host 侧）
  // ============================
  // 获取 FullStructureData
  auto& md_data = DataManager::getInstance().getMDData();
  if (!md_data || !md_data->_structure_data) {
    std::cerr << "[SpecialBondBuilder] Error: MDData or structure_data is null"
              << std::endl;
    return;
  }

  auto full =
      std::dynamic_pointer_cast<FullStructureData>(md_data->_structure_data);
  if (!full) {
    std::cerr << "[SpecialBondBuilder] Error: Cannot cast to FullStructureData"
              << std::endl;
    return;
  }

  const rbmd::Id nmax =
      static_cast<rbmd::Id>(nlocal + ghost_atom_ids.size());

  // 释放旧的 special 数据（如果存在）
  if (full->_h_nspecial) {
    std::free(full->_h_nspecial);
    full->_h_nspecial = nullptr;
  }
  if (full->_h_special) {
    std::free(full->_h_special);
    full->_h_special = nullptr;
  }

  // 分配新的 special 数据（nmax * 3 和 nmax * maxspecial）
  const std::size_t nspecial_size = static_cast<std::size_t>(nmax) * 3;
  const std::size_t special_size =
      static_cast<std::size_t>(nmax) * global_maxspecial;

  full->_h_nspecial =
      static_cast<int*>(std::malloc(nspecial_size * sizeof(int)));
  full->_h_special =
      static_cast<rbmd::Id*>(std::malloc(special_size * sizeof(rbmd::Id)));

  if (!full->_h_nspecial || !full->_h_special) {
    std::cerr
        << "[SpecialBondBuilder] Error: Failed to allocate host special arrays"
        << std::endl;
    if (full->_h_nspecial) {
      std::free(full->_h_nspecial);
      full->_h_nspecial = nullptr;
    }
    if (full->_h_special) {
      std::free(full->_h_special);
      full->_h_special = nullptr;
    }
    return;
  }

  // 初始化为 0
  std::memset(full->_h_nspecial, 0, nspecial_size * sizeof(int));
  std::memset(full->_h_special, 0, special_size * sizeof(rbmd::Id));

  // 拷贝本地原子的数据到 FullStructureData
  // nspecial: 只拷贝本地原子部分（nlocal * 3）
  std::memcpy(full->_h_nspecial, h_nspecial.data(), nlocal * 3 * sizeof(int));

  // special: 只拷贝本地原子部分
  for (std::size_t tid = 0; tid < nlocal; ++tid) {
    const int n14 = h_nspecial[tid * 3 + 2];
    const std::size_t copy_count =
        std::min(static_cast<std::size_t>(n14),
                 static_cast<std::size_t>(global_maxspecial));

    for (std::size_t k = 0; k < copy_count; ++k) {
      full->_h_special[tid * global_maxspecial + k] =
          h_special[tid * global_maxspecial + k];
    }
  }

  // 设置元数据
  full->_h_maxspecial = global_maxspecial;
  full->_h_nmax_special = nmax;

  // 同时更新 DeviceData 的 maxspecial 和 nmax（为后续 H2D 做准备）
  if (_device_data) {
    _device_data->maxspecial = global_maxspecial;
    _device_data->nmax = nmax;
  }

#ifdef READER_ENABLE_MPI
  if (my_rank == 0) {
    std::cout << "[SpecialBondBuilder] Host special data prepared: nmax="
              << nmax << ", maxspecial=" << global_maxspecial << std::endl;
  }
#else
  std::cout << "[SpecialBondBuilder] Host special data prepared: nmax=" << nmax
            << ", maxspecial=" << global_maxspecial << std::endl;
#endif

  // ============================
  // T022: 一致性校验（debug 模式）
  // ============================
#if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
  // 计算 checksum（用于一致性验证）
  const uint64_t local_checksum = ComputeSpecialChecksum(
      full->_h_nspecial, full->_h_special, nlocal, global_maxspecial);

#ifdef READER_ENABLE_MPI
  std::cout << "[SpecialBondBuilder] Rank " << my_rank << " checksum: 0x"
            << std::hex << local_checksum << std::dec << std::endl;

  // 收集所有 rank 的 checksum 到 rank 0 进行对比
  std::vector<uint64_t> all_checksums;
  if (my_rank == 0) {
    all_checksums.resize(num_ranks);
  }

  MPI_Gather(&local_checksum, 1, MPI_UNSIGNED_LONG_LONG,
             all_checksums.empty() ? nullptr : all_checksums.data(), 1,
             MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::cout << "[SpecialBondBuilder] All rank checksums:" << std::endl;
    for (int r = 0; r < num_ranks; ++r) {
      std::cout << "  Rank " << r << ": 0x" << std::hex << all_checksums[r]
                << std::dec << std::endl;
    }
  }
#else
  std::cout << "[SpecialBondBuilder] Checksum: 0x" << std::hex << local_checksum
            << std::dec << std::endl;
#endif
#endif  // DEBUG
}

uint64_t reader::mpi::SpecialBondBuilder::ComputeSpecialChecksum(
    const int* h_nspecial, const rbmd::Id* h_special, std::size_t nlocal,
    int maxspecial) const {
  // T022: 使用 FNV-1a hash 算法计算 checksum
  // 对齐 LAMMPS 的一致性验证语义
  constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
  constexpr uint64_t FNV_PRIME = 1099511628211ULL;

  uint64_t hash = FNV_OFFSET_BASIS;

  // Hash nspecial 数据
  for (std::size_t tid = 0; tid < nlocal; ++tid) {
    for (int k = 0; k < 3; ++k) {
      const int val = h_nspecial[tid * 3 + k];
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&val);
      for (std::size_t b = 0; b < sizeof(int); ++b) {
        hash ^= bytes[b];
        hash *= FNV_PRIME;
      }
    }
  }

  // Hash special 数据（只 hash 有效部分：nspecial[2] 指示的范围）
  for (std::size_t tid = 0; tid < nlocal; ++tid) {
    const int n14 = h_nspecial[tid * 3 + 2];
    const int valid_count = std::min(n14, maxspecial);

    for (int k = 0; k < valid_count; ++k) {
      const rbmd::Id val = h_special[tid * maxspecial + k];
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&val);
      for (std::size_t b = 0; b < sizeof(rbmd::Id); ++b) {
        hash ^= bytes[b];
        hash *= FNV_PRIME;
      }
    }
  }

  return hash;
}

void reader::mpi::SpecialBondBuilder::Build(
    std::vector<rbmd::Id> local_atom_ids, std::vector<rbmd::Id> ghost_atom_ids,
    std::vector<std::tuple<rbmd::Id, rbmd::Id>> local_bonds,
    const std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id>>&
        local_angles,
    const std::vector<std::tuple<int, rbmd::Id, rbmd::Id, rbmd::Id, rbmd::Id>>&
        local_dihedrals,
    int my_rank, int num_ranks) {
  (void)BuildAdjacencyMaps(std::move(local_atom_ids), std::move(ghost_atom_ids),
                           std::move(local_bonds), local_angles,
                           local_dihedrals, my_rank, num_ranks);
}
