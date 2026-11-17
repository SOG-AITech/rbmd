#pragma once
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define RBMD_FIXEDP_HD __host__ __device__ __forceinline__
#else
#define RBMD_FIXEDP_HD inline
#endif

namespace rbmd { namespace fixedp {
// SplitMix64 is a reproducible counter stream, not an IID proof. No rank or
// cache-build index enters the stream: owner global ID + force epoch determine it.
RBMD_FIXEDP_HD std::uint64_t Mix(std::uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
struct Counter {
  std::uint64_t state;
  RBMD_FIXEDP_HD Counter(std::uint64_t owner, std::uint64_t epoch)
      : state(Mix(owner + UINT64_C(0xd1b54a32d192ed03)) ^
              Mix(epoch + UINT64_C(0x9e3779b97f4a7c15)) ^
              UINT64_C(0x4649584544505253)) {}
  RBMD_FIXEDP_HD std::uint64_t operator()() {
    state += UINT64_C(0x9e3779b97f4a7c15);
    return Mix(state);
  }
};
template <typename Generator>
RBMD_FIXEDP_HD bool UniformBelow(std::uint64_t bound, Generator& next,
                                std::uint64_t& value) {
  if (bound == 0) return false;
  // Unsigned wrap is defined; rejecting the short prefix removes modulo bias.
  const std::uint64_t threshold = (std::uint64_t(0) - bound) % bound;
  std::uint64_t draw;
  do { draw = next(); } while (draw < threshold);
  value = draw % bound;
  return true;
}
template <typename Id>
RBMD_FIXEDP_HD bool LessIndex(Id a, Id b, const Id* global_ids) {
  return global_ids[a] < global_ids[b] ||
      (global_ids[a] == global_ids[b] && a < b);
}
template <typename Id>
RBMD_FIXEDP_HD void Sift(Id* row, std::uint64_t count,
                        std::uint64_t root, const Id* global_ids) {
  while (count > 1 && root <= (count - 2) / 2) {
    std::uint64_t child = root * 2 + 1;
    if (child + 1 < count && LessIndex(row[child], row[child+1], global_ids)) ++child;
    if (!LessIndex(row[root], row[child], global_ids)) return;
    const Id tmp = row[root]; row[root] = row[child]; row[child] = tmp;
    root = child;
  }
}
template <typename Id>
RBMD_FIXEDP_HD void SortRow(Id* row, std::uint64_t count, const Id* global_ids) {
  for (std::uint64_t root = count / 2; root > 0; --root)
    Sift(row, count, root - 1, global_ids);
  for (std::uint64_t end = count; end > 1; --end) {
    const Id tmp = row[0]; row[0] = row[end-1]; row[end-1] = tmp;
    Sift(row, end-1, 0, global_ids);
  }
}
// Called exactly once for each UNIQUE current physical-shell global ID.
// count is the running physical M, not the full candidate or core+skin count.
template <typename Id, typename Generator>
RBMD_FIXEDP_HD bool ReservoirAccept(Id candidate, std::uint64_t count,
    Id requested, Id* output, std::uint64_t stride, std::uint64_t owner,
    Generator& next) {
  if (requested <= 0 || count == 0) return false;
  std::uint64_t slot = count - 1;
  if (count > static_cast<std::uint64_t>(requested) &&
      !UniformBelow(count, next, slot)) return false;
  if (slot < static_cast<std::uint64_t>(requested))
    output[slot * stride + owner] = candidate;
  return true;
}
}}  // namespace rbmd::fixedp
#undef RBMD_FIXEDP_HD
