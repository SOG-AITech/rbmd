#include "common/revision_options.h"
#include "common/mpi_root_guard.hpp"
#include <thrust/host_vector.h>
#include <fstream>
#include <iomanip>

namespace {
void RecordBenchmarkRblSamples(const NeighborList& list, rbmd::Id n, rbmd::Id step) {
  const auto& options = rbmd::revision::GetOptions();
  // Accuracy-only reads must never enter the production timing path.
  if (!options.forces || !options.fixed_samples || step <= 0 || step % options.interval) return;
  if (list._rbl_fused_sampling || list._d_random_neighbor_num.size() < static_cast<std::size_t>(n))
    throw std::runtime_error("RBL sample diagnostics need a materialized owned-atom sample list");
  thrust::host_vector<rbmd::Id> counts(list._d_random_neighbor_num.begin(), list._d_random_neighbor_num.begin()+n);
  long double sum=0, square=0;
  rbmd::Id lo=n ? counts[0] : 0, hi=0;
  for (auto count : counts) {
    sum+=count; square+=static_cast<long double>(count)*count;
    lo=std::min(lo,count); hi=std::max(hi,count);
  }
  static std::ofstream out(options.prefix+".rbl_samples.rank"+std::to_string(rbmd::mpi::CurrentRank())+".csv");
  static bool header=false;
  if (!header) { out << "sample,n_owned,sum,sum_square,min,max\n"; header=true; }
  out << std::setprecision(20) << step << ',' << n << ',' << sum << ',' << square << ',' << lo << ',' << hi << '\n';
  out.flush();
  if (!out) throw std::runtime_error("Cannot write RBL sample diagnostics");
}
}
