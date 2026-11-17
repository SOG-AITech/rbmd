#include "core/replicate_atoms.h"
#include <cassert>
#include <iostream>
using namespace reader;
using namespace reader::replication;
int main() {
 ParsedSnapshot seed;seed.box={0,10,0,10,0,10};
 ParsedSnapshot::Box big{0,30,0,20,0,10};
 ParsedSnapshot::Atom a,b;a.id=2;b.id=9;a.x=9.8;b.x=.2;a.y=b.y=4;a.z=b.z=4;
 const std::array<std::size_t,3> factors{3,2,1};
 for (std::size_t t=0;t<6;++t) {
  auto u=BondTargetReplica(seed.box,a,b,t,factors);
  assert(u==((t/3)*3+(t%3+1)%3));
  assert(BondTargetReplica(seed.box,b,a,u,factors)==t);
  auto ra=BuildReplicatedAtom(seed.box,big,a,t,10,0,(t%3)*10,(t/3)*10,0,true);
  auto rb=BuildReplicatedAtom(seed.box,big,b,u,10,0,(u%3)*10,(u/3)*10,0,true);
  double dx=rb.x-ra.x;dx-=30*std::round(dx/30);assert(std::abs(dx-.4)<2e-6);
 }
 // Chain 8 -> 1 -> 4 -> 7: endpoint must follow atom3, not its nearest
 // image relative to atom2. Each consecutive bond remains length 3.
 ParsedSnapshot::Atom chain[4];double xs[]={8,1,4,7};
 std::size_t tiles[4]={0,0,0,0};
 for(int i=0;i<4;++i){chain[i].x=xs[i];chain[i].y=chain[i].z=1;}
 for(int i=1;i<4;++i)tiles[i]=BondTargetReplica(seed.box,chain[i-1],chain[i],tiles[i-1],factors);
 assert(tiles[1]==1 && tiles[2]==1 && tiles[3]==1);
 assert(BondTargetReplica(seed.box,chain[1],chain[3],tiles[1],factors)!=tiles[3]);
 for(int i=1;i<4;++i)assert(xs[i]+10*tiles[i]-(xs[i-1]+10*tiles[i-1])==3);
 // Streaming local atom enumeration must produce identical atom IDs/positions.
 seed.atoms={a,b};
 for(int rank=0;rank<3;++rank){int count=0;
  ForEachLocalReplica(seed,big,factors,{3,1,1},{rank,0,0},10,true,rank,
   [](const auto& atom){return int(atom.x/10);},[&](const auto& atom){assert(int(atom.x/10)==rank);++count;});
  assert(count==4);
 }
 // Half-box ties and an unreplicated axis retain ordinary minimum images.
 b.x=a.x-5;assert(BondTargetReplica(seed.box,a,b,0,factors)==0);
 a.z=9.8;b.z=.2;assert(BondTargetReplica(seed.box,a,b,0,factors)==0);
 std::cout<<"periodic bond, winding dihedral, and streaming ownership PASS\n";
}
