// Compile with hipcc -x hip or nvcc -x cu, USE_64BIT_IDS=1. Uses the same
// device predicate as SelectedTrajectoryOutput, including IDs above 2^32.
#include <cassert>
#include <iostream>
#include <vector>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include "output/include/trajectory_selection.h"
using namespace rbmd;
using namespace rbmd::trajectory;

template<class T> T* ptr(thrust::device_vector<T>& v) {
  return thrust::raw_pointer_cast(v.data());
}
int main() {
  static_assert(sizeof(Id) == 8);
  std::vector<Id> hi{9999999999LL, 4294967296LL, 3, 12};
  std::vector<Id> ht{2,2,1,2};
  std::vector<Real> hx{0,1,1,11}, hy{0,0,0,0};
  thrust::device_vector<Id> ids=hi, types=ht;
  thrust::device_vector<Real> x=hx,y=hy,z=hy;
  auto options=ParseOptions(nlohmann::ordered_json::parse(R"({"selection":{"types":[3],"region":{"x":[0,2]}}})"));
  thrust::device_vector<Id> wanted_types=options.types;
  thrust::device_vector<Region> regions=options.regions;
  Predicate p;
  p.atom_ids=ptr(ids);p.atom_types=ptr(types);p.x=ptr(x);p.y=ptr(y);p.z=ptr(z);
  p.types=ptr(wanted_types);p.num_types=1;p.regions=ptr(regions);p.num_regions=1;
  thrust::device_vector<size_t> indices(4);
  auto first=thrust::make_counting_iterator<size_t>(0);
  auto end=thrust::copy_if(thrust::device,first,first+4,indices.begin(),p);
  assert(end-indices.begin()==2);
  thrust::device_vector<Id> packed(2);
  thrust::gather(thrust::device,indices.begin(),end,ids.begin(),packed.begin());
  std::vector<Id> selected(2);
  thrust::copy(packed.begin(),packed.end(),selected.begin());
  assert(selected[0]==9999999999LL && selected[1]==4294967296LL);
  // Freeze identities, then simulate migration/reordering and region exit.
  std::sort(selected.begin(),selected.end());
  thrust::device_vector<Id> frozen=selected;
  hi={12,9999999999LL,3,4294967296LL};ids=hi;
  hx={11,100,1,-100};x=hx;
  p.atom_ids=ptr(ids);p.x=ptr(x);p.ids=ptr(frozen);p.num_ids=2;
  p.filter_ids=true;p.num_regions=0;p.num_types=0;
  end=thrust::copy_if(thrust::device,first,first+4,indices.begin(),p);
  assert(end-indices.begin()==2);
  std::vector<size_t> positions(2);
  thrust::copy(indices.begin(),end,positions.begin());
  assert(positions[0]==1 && positions[1]==3);
  p.num_ids=0;
  assert(thrust::copy_if(thrust::device,first,first+4,indices.begin(),p)==indices.begin());
  std::cout<<"64-bit GPU selection passed: region, type, compaction, frozen IDs, migration, empty cohort; Real="<<sizeof(Real)<<"\n";
}
