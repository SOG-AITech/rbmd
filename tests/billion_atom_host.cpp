// Production sparse lookup and spatial replication, without a GPU/MPI SDK.
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include "common/types.h"
#include "common/atom_index_map.h"
#include "core/replicate_atoms.h"
using namespace reader;
using namespace reader::replication;

void CheckMaps() {
  rbmd::Id keys[] = {2, 11, 11, 10000000001LL, 10000000009LL};
  rbmd::LocalAtomIndex indices[] = {3, 0, 4, 2, 1};
  rbmd::AtomIndexMapView map{keys, indices, 5};
  assert(map[11] == 0 && map[10000000001LL] == 2);
  assert(map[10000000009LL] == 1 && map[0] == -1 && map[-1] == -1);
  assert(map[10000000010LL] == -1 && rbmd::AtomIndexMapView{}[0] == -1);
  const rbmd::NeighborOffset begin = 2200000000LL;
  rbmd::NeighborOffset visited = 0;
  for (rbmd::NeighborOffset i = begin; i < begin + 7; ++i) ++visited;
  assert(visited == 7);
}

void CheckReplication() {
  ParsedSnapshot seed;
  seed.box = {-1, 1, 2, 5, -2, 2};
  for (int i = 0; i < 5; ++i) {
    ParsedSnapshot::Atom atom;
    atom.id = i * 3; atom.molecule = -1;
    atom.x = -1 + i * 0.4; atom.y = 2 + i * 0.5; atom.z = -2 + i * 0.7;
    atom.vx = i + 0.25; atom.ix = i - 2; atom.iy = 2 - i; atom.iz = i % 2;
    seed.atoms.push_back(atom);
  }
  // Out-of-cell coordinates and image flags must preserve the serial result.
  seed.atoms[1].x += 6;
  seed.atoms[2].y -= 9;
  seed.atoms[3].z = seed.box.zhi;
  const std::array<std::size_t,3> factors{5, 4, 3};
  ParsedSnapshot::Box box{-1, 9, 2, 14, -2, 10};
  for (bool periodic : {false, true}) {
    std::map<rbmd::Id, ParsedSnapshot::Atom> reference;
    for (std::size_t z=0; z<3; ++z) for (std::size_t y=0; y<4; ++y)
      for (std::size_t x=0; x<5; ++x) for (const auto& a : seed.atoms) {
        auto b = BuildReplicatedAtom(seed.box, box, a, (z*4+y)*5+x,
                                    13, 0, x*2, y*3, z*4, periodic);
        reference.emplace(b.id, b);
      }
    // Includes thin and empty partitions, and dimensions with only one rank.
    for (auto dims : {std::array<int,3>{1,1,1}, {2,3,2}, {11,1,1}, {1,2,1}}) {
      auto owner = [&](const ParsedSnapshot::Atom& a) {
        int x = std::clamp(int((a.x-box.xlo)*dims[0]/(box.xhi-box.xlo)),0,dims[0]-1);
        int y = std::clamp(int((a.y-box.ylo)*dims[1]/(box.yhi-box.ylo)),0,dims[1]-1);
        int z = std::clamp(int((a.z-box.zlo)*dims[2]/(box.zhi-box.zlo)),0,dims[2]-1);
        return (x*dims[1]+y)*dims[2]+z;
      };
      std::set<rbmd::Id> seen;
      for(int x=0;x<dims[0];++x) for(int y=0;y<dims[1];++y) for(int z=0;z<dims[2];++z) {
        ForEachLocalReplica(seed,box,factors,dims,{x,y,z},13,periodic,
            (x*dims[1]+y)*dims[2]+z,owner,[&](const auto& a) {
              assert(seen.insert(a.id).second);
              const auto& b=reference.at(a.id);
              assert(a.x==b.x && a.y==b.y && a.z==b.z && a.vx==b.vx);
              assert(a.ix==b.ix && a.iy==b.iy && a.iz==b.iz);
            });
      }
      assert(seen.size()==reference.size());
    }
  }
  // Address a cell at the far end of a 10-billion-atom virtual lattice without
  // allocating or visiting that lattice. Only a few local candidates are made.
  seed.atoms.resize(4);
  for (int i=0;i<4;++i) {
    seed.atoms[i] = {};
    seed.atoms[i].id=i; seed.atoms[i].molecule=-1;
    seed.atoms[i].x=seed.atoms[i].y=seed.atoms[i].z=0.1f*(i+1);
  }
  seed.box={0,1,0,1,0,1};
  int count=0;
  ForEachLocalReplica(seed,{0,5000,0,5000,0,100},{5000,5000,100},
      {5000,5000,100},{4999,4999,99},4,false,1,
      [](const auto& a) {return a.x>=4999 && a.y>=4999 && a.z>=99 ? 1:0;},
      [&](const auto& a) {assert(a.id>=9999999996LL && a.id<10000000000LL);++count;});
  assert(count==4);
  bool overflow=false;
  try { CheckedIdAdd(1, std::numeric_limits<rbmd::Id>::max(), "test"); }
  catch(const std::overflow_error&) {overflow=true;}
  assert(overflow);
}
int main() {
  CheckMaps(); CheckReplication();
  std::cout << "PASS " << (sizeof(rbmd::Real)==8 ? "double" : "float")
            << ": sparse IDs > 10B, duplicate/missing IDs, 64-bit offsets, local replication, images, velocities, virtual 10B lattice\n";
}
