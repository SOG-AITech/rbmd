#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "core/parsed_snapshot.h"

namespace reader::mpi {

struct AtomPartition {
  std::vector<rbmd::Id> offsets;
  std::vector<rbmd::Id> atom_ids;
  std::vector<int> owners;
};

struct StreamCtx {
  ParsedSnapshot* snapshot = nullptr;
  AtomPartition partition;
  int rank = 0;
  int size = 1;
  bool is_root = false;
  bool active = false;
  std::unordered_map<rbmd::Id, std::size_t> local_index;
#ifdef READER_ENABLE_MPI
  struct Impl;
  std::shared_ptr<Impl> impl;
#endif
};

bool IsEnabled();
int Rank();
int Size();
void DistributeSnapshot(ParsedSnapshot& snapshot);

StreamCtx BeginStreaming(ParsedSnapshot& snapshot, rbmd::Id expected_atoms);
void ReceiveLoop(StreamCtx& ctx);
void EndStreaming(StreamCtx& ctx);
void StreamAtom(StreamCtx& ctx, const ParsedSnapshot::Atom& atom);
void StreamVelocity(StreamCtx& ctx,
                    rbmd::Id atom_id,
                    rbmd::Real vx,
                    rbmd::Real vy,
                    rbmd::Real vz);
void StreamBond(StreamCtx& ctx, const ParsedSnapshot::Bond& bond);
void StreamAngle(StreamCtx& ctx, const ParsedSnapshot::Angle& angle);
void StreamDihedral(StreamCtx& ctx, const ParsedSnapshot::Dihedral& dihedral);
void StreamImproper(StreamCtx& ctx, const ParsedSnapshot::Improper& improper);

}  // namespace reader::mpi
