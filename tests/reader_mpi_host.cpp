#include <cassert>
#include <thread>
#include <set>
#include <iostream>
#include <sys/mman.h>
#include "common/mpi_chunked.h"
#include "../src/reader/src/mpi/mpi_distribution.cpp"
void CreateMpiFullAtomDataPacket(MPI_Datatype*) {std::abort();}
int main() {
  using namespace reader;
  std::vector<ParsedSnapshot> snapshots(4);
  ParsedSnapshot seed;
  seed.box={0,1,0,1,0,1};seed.topology.atom_types=1;seed.masses[0]=63.546;
  for(int i=0;i<4;++i) {
    ParsedSnapshot::Atom a;a.id=10000000000LL+i;a.molecule=-1;
    a.x=a.y=a.z=rbmd::Real(0.1*(i+1));a.vx=i+0.5;seed.atoms.push_back(a);
  }
  snapshots[0]=seed;snapshots[0].box={0,8,0,2,0,1};snapshots[0].atoms.clear();
  std::vector<std::thread> threads;
  for(int r=0;r<4;++r)threads.emplace_back([&,r] {
    mock_mpi::rank=r;
    auto ctx=mpi::BeginStreaming(snapshots[r],r==0?64:0,r==0?&seed:nullptr,{8,2,1},false);
    assert(ctx.generated_locally && !ctx.impl);
    mpi::ReceiveLoop(ctx); // Must skip receive loop for generated atoms.
    mpi::EndStreaming(ctx);
    assert(ctx.partition.atom_ids.empty() && ctx.partition.owners.empty());
    assert(ctx.partition.offsets.back()==64 && snapshots[r].atoms.size()==16);
    assert(snapshots[r].masses.at(0)==seed.masses.at(0));
  });
  for(auto& t:threads)t.join();
  std::set<rbmd::Id> ids;
  for(int r=0;r<4;++r)for(const auto& a:snapshots[r].atoms) {
    assert(a.x>=r*2 && a.x<(r+1)*2);assert(ids.insert(a.id).second);
    assert(a.vx>=0.5 && a.vx<=3.5);
  }
  assert(ids.size()==64);
  // Virtual reservation tests element counts > INT_MAX with no physical payload.
  const std::size_t count=static_cast<std::size_t>(2147483647)+19;
  const auto bytes=count*sizeof(double);
  auto* data=static_cast<double*>(mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0));
  assert(data!=MAP_FAILED);
  std::vector<MPI_Request> requests;
  rbmd::mpi::IrecvChunks(data,count,MPI_DOUBLE,2,77,MPI_COMM_WORLD,requests);
  std::size_t seen=0;
  for(const auto& m:mock_mpi::messages) {
    assert(m.count>0 && static_cast<std::size_t>(m.count)<=rbmd::mpi::kChunkElements);
    assert(m.data==data+seen && m.peer==2 && m.tag==77 && !m.send);seen+=m.count;
  }
  assert(seen==count);rbmd::mpi::WaitChunks(requests);
  mock_mpi::messages.clear();requests.clear();
  rbmd::mpi::IsendChunks(data,count,MPI_DOUBLE,3,78,MPI_COMM_WORLD,requests);
  seen=0;
  for(const auto& m:mock_mpi::messages){assert(m.data==data+seen && m.peer==3 && m.tag==78 && m.send);seen+=m.count;}
  assert(seen==count);rbmd::mpi::WaitChunks(requests);
  munmap(data,bytes);
  std::cout<<"PASS: production MPI reader with 4 simulated ranks, sparse seed IDs, no global ownership arrays, chunk offsets/counts > INT_MAX\n";
}
