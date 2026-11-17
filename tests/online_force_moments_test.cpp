#include "common/reference_force.h"
#include <cassert>
#include <sstream>
#include <vector>
#include <iostream>
#include <random>
using namespace rbmd::revision;
int main() {
  OnlineForceMoments stats;
  std::vector<std::array<double,3>> errors{{{1,2,3}},{{3,4,5}},{{5,6,7}}};
  double mse=0;
  for (const auto& e:errors) { stats.Add(e); for(double x:e) mse+=x*x; }
  mse/=errors.size();
  assert(stats.samples==3);
  assert(stats.BiasSquared()==50);
  assert(stats.m2==24);
  assert(std::abs(mse-(stats.BiasSquared()+stats.m2/3))<1e-12);
  assert(stats.m2/(stats.samples-1)==12);
  OnlineForceMoments constant;
  for(int i=0;i<1000;++i) constant.Add({{1e12,-1e12,1e12}});
  assert(constant.m2==0);
  OnlineForceMoments stable;
  stable.Add({{1e12-1,0,0}});stable.Add({{1e12+1,0,0}});
  assert(stable.m2==2);
  // Independent two-pass reduction, with a large nonzero mean and uneven samples.
  std::mt19937 rng(42);
  std::normal_distribution<double> normal(0, 3);
  for (int atom=0; atom<17; ++atom) {
    OnlineForceMoments online;
    std::vector<std::array<double,3>> samples(73);
    std::array<double,3> mean{};
    for (auto& row:samples) for (int d=0; d<3; ++d) {
      row[d]=10000*atom+normal(rng);
      mean[d]+=row[d]/samples.size();
    }
    double two_pass=0;
    for (const auto& row:samples) {
      online.Add(row);
      for (int d=0; d<3; ++d) two_pass+=(row[d]-mean[d])*(row[d]-mean[d]);
    }
    assert(std::abs(online.m2-two_pass)<1e-7);
  }
  ReferenceFrame box;box.lower={{-5,0,0}};box.length={{10,10,10}};
  assert(ReferencePositionMatches(box,0,5,-5,1e-6));
  assert(!ReferencePositionMatches(box,0,5.01,-5,1e-6));
  std::istringstream dump("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id x y z fx fy fz\n1 0 0 0 1 2 3\n2 1 0 0 3 4 5\n3 2 0 0 5 6 7\n");
  std::vector<ReferenceForce> owned;
  auto frame=VisitReferenceDump(dump,0,-1,[&](std::int64_t id,const ReferenceForce& r){if(id==1)owned.push_back(r);});
  assert(frame.count==3 && owned.size()==1 && owned[0].force[0]==3);
  // A repeated/reordered dump must not silently corrupt streaming ownership checks.
  std::istringstream bad("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id x y z fx fy fz\n1 0 0 0 1 2 3\n1 0 0 0 1 2 3\n");
  bool rejected=false;
  try { VisitReferenceDump(bad,0,0,[](std::int64_t,const ReferenceForce&){}); }
  catch(const std::exception&){rejected=true;}
  assert(rejected);
  std::cout << "online moments, streaming ownership filter, periodic match: PASS\n";
}
