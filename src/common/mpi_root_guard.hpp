#pragma once

#include <cstdio>
#include <iostream>
#include <streambuf>
#include <unistd.h>

#if defined(USE_MPI) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <mpi.h>
#define RBMD_HOST_MPI_AVAILABLE 1
#else
#define RBMD_HOST_MPI_AVAILABLE 0
#endif

namespace rbmd::mpi {

namespace detail {

inline std::streambuf*& SavedCoutBuffer() {
  static std::streambuf* buffer = nullptr;
  return buffer;
}

inline std::streambuf*& SavedCerrBuffer() {
  static std::streambuf* buffer = nullptr;
  return buffer;
}

inline std::streambuf*& SavedClogBuffer() {
  static std::streambuf* buffer = nullptr;
  return buffer;
}

inline int& SavedStdoutFd() {
  static int fd = -1;
  return fd;
}

inline int& SavedStderrFd() {
  static int fd = -1;
  return fd;
}

inline bool& StreamsSuppressed() {
  static bool suppressed = false;
  return suppressed;
}

}  // namespace detail

inline int CurrentRank() {
#if RBMD_HOST_MPI_AVAILABLE
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      int current_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
      return current_rank;
    }
  }
#endif
  return 0;
}

inline bool ShouldWriteRootOnlyOutput() { return CurrentRank() == 0; }

class NullStreamBuffer final : public std::streambuf {
 protected:
  int overflow(int ch) override { return traits_type::not_eof(ch); }
};

inline void SuppressNonRootOutputStreams() {
#if RBMD_HOST_MPI_AVAILABLE
  if (ShouldWriteRootOnlyOutput()) {
    return;
  }

  if (detail::StreamsSuppressed()) {
    return;
  }

  static NullStreamBuffer null_buffer;
  detail::SavedCoutBuffer() = std::cout.rdbuf();
  detail::SavedCerrBuffer() = std::cerr.rdbuf();
  detail::SavedClogBuffer() = std::clog.rdbuf();
  detail::SavedStdoutFd() = ::dup(fileno(stdout));
  detail::SavedStderrFd() = ::dup(fileno(stderr));
  std::cout.rdbuf(&null_buffer);
  std::cerr.rdbuf(&null_buffer);
  std::clog.rdbuf(&null_buffer);

  std::fflush(stdout);
  std::fflush(stderr);
  std::freopen("/dev/null", "w", stdout);
  std::freopen("/dev/null", "w", stderr);
  detail::StreamsSuppressed() = true;
#endif
}

inline void RestoreSuppressedOutputStreams() {
#if RBMD_HOST_MPI_AVAILABLE
  if (!detail::StreamsSuppressed()) {
    return;
  }

  std::fflush(stdout);
  std::fflush(stderr);

  if (detail::SavedStdoutFd() >= 0) {
    ::dup2(detail::SavedStdoutFd(), fileno(stdout));
  }
  if (detail::SavedStderrFd() >= 0) {
    ::dup2(detail::SavedStderrFd(), fileno(stderr));
  }

  if (detail::SavedCoutBuffer() != nullptr) {
    std::cout.rdbuf(detail::SavedCoutBuffer());
  }
  if (detail::SavedCerrBuffer() != nullptr) {
    std::cerr.rdbuf(detail::SavedCerrBuffer());
  }
  if (detail::SavedClogBuffer() != nullptr) {
    std::clog.rdbuf(detail::SavedClogBuffer());
  }

  detail::StreamsSuppressed() = false;
#endif
}

}  // namespace rbmd::mpi
#undef RBMD_HOST_MPI_AVAILABLE
