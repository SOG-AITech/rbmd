#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

namespace rbmd::mpi {
inline void CheckChunkMpi(int status) {
  if (status != MPI_SUCCESS) throw std::runtime_error("Chunked MPI transfer failed");
}
// Same source/tag messages are matched in posting order. The limit is in
// datatype elements, not bytes; each chunk fits legacy MPI's signed int count.
inline constexpr std::size_t kChunkElements = 1u << 26;
template<class T>
void IrecvChunks(T* data, std::size_t count, MPI_Datatype type, int peer,
                 int tag, MPI_Comm comm, std::vector<MPI_Request>& requests) {
  for (std::size_t offset = 0; offset < count;) {
    const auto n = std::min(kChunkElements, count - offset);
    MPI_Request request = MPI_REQUEST_NULL;
    CheckChunkMpi(MPI_Irecv(data + offset, static_cast<int>(n), type, peer,
                            tag, comm, &request));
    requests.push_back(request);
    offset += n;
  }
}
template<class T>
void IsendChunks(const T* data, std::size_t count, MPI_Datatype type, int peer,
                 int tag, MPI_Comm comm, std::vector<MPI_Request>& requests) {
  for (std::size_t offset = 0; offset < count;) {
    const auto n = std::min(kChunkElements, count - offset);
    MPI_Request request = MPI_REQUEST_NULL;
    CheckChunkMpi(MPI_Isend(data + offset, static_cast<int>(n), type, peer,
                            tag, comm, &request));
    requests.push_back(request);
    offset += n;
  }
}
inline void WaitChunks(std::vector<MPI_Request>& requests) {
  for (std::size_t offset = 0; offset < requests.size();) {
    const auto n = std::min(kChunkElements, requests.size() - offset);
    CheckChunkMpi(MPI_Waitall(static_cast<int>(n), requests.data() + offset,
                              MPI_STATUSES_IGNORE));
    offset += n;
  }
}
}  // namespace rbmd::mpi
