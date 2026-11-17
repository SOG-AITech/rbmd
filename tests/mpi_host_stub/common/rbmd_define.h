#pragma once
#include "common/types.h"
#include <mpi.h>
#define MPI_RBMD_ID MPI_LONG_LONG
#if USE_DOUBLE
#define MPI_RBMD_REAL MPI_DOUBLE
#else
#define MPI_RBMD_REAL MPI_FLOAT
#endif
