#include <cstdlib>
#include <iostream>

#include "application/md_application.h"
#include "common/mpi_root_guard.hpp"
#ifdef USE_MPI
#include "rbmd_parallel_until_locator.h"
#include "mpi.h"
#endif

int main(int argc, char* argv[]) {
  //CHECK_RUNTIME(hipSetDevice(1));
#ifdef USE_MPI
  auto parallel_until =
      RbmdParallelUntilLocator::GetInstance().GetRbmdParallelUntil();
  parallel_until->Init(CollectiveCommunicator::Backend::MPI);
  const char* keep_all_rank_stdio = std::getenv("RBMD_DEBUG_KEEP_ALL_RANK_STDIO");
  if (keep_all_rank_stdio == nullptr || keep_all_rank_stdio[0] == '\0' ||
      keep_all_rank_stdio[0] == '0') {
    rbmd::mpi::SuppressNonRootOutputStreams();
  }
#endif

  try {
    std::shared_ptr<Application> app =
        std::make_shared<MDApplication>(argc, argv);
    app->Run();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
  } catch (const std::exception& e) {
#ifdef USE_MPI
    rbmd::mpi::RestoreSuppressedOutputStreams();
#endif
    std::cerr << e.what() << std::endl;
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
#endif
    return EXIT_FAILURE;
  }
}
