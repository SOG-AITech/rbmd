// Generate >2 GiB through an istream without storing a frame or creating a large file.
#include "common/reference_force.h"
#include <streambuf>
#include <cassert>
#include <iostream>
#include <sys/resource.h>
class GeneratedDump : public std::streambuf {
 public:
  static constexpr std::int64_t count=2100000;
  std::uint64_t bytes=0;
 private:
  std::int64_t row=-1;
  std::string buffer;
  int_type underflow() override {
    if (gptr() && gptr()<egptr()) return traits_type::to_int_type(*gptr());
    if(row==count) return traits_type::eof();
    if(row<0) buffer="ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n"+std::to_string(count)+"\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id x y z fx fy fz padding\n";
    else buffer=std::to_string(row+1)+" 0 0 0 1 2 3 "+std::string(1024,'x')+"\n";
    ++row;bytes+=buffer.size();setg(buffer.data(),buffer.data(),buffer.data()+buffer.size());
    return traits_type::to_int_type(*gptr());
  }
};
int main(){
  GeneratedDump buffer;std::istream input(&buffer);
  std::int64_t seen=0;double sum=0;
  auto frame=rbmd::revision::VisitReferenceDump(input,0,-1,[&](std::int64_t id,const rbmd::revision::ReferenceForce& r){assert(id==seen++);sum+=r.force[0];});
  assert(seen==buffer.count && frame.count==seen && sum==seen);
  assert(buffer.bytes>std::uint64_t(2147483647));
  rusage usage{};getrusage(RUSAGE_SELF,&usage);
  std::cout<<"PASS bytes="<<buffer.bytes<<" atoms="<<seen<<" maxrss="<<usage.ru_maxrss<<"\n";
}
