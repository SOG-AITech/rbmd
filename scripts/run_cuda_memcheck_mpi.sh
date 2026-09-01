#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run an rbmd CUDA build under Compute Sanitizer memcheck and archive one report
per MPI rank/process.

Required:
  RBMD_BIN=/absolute/path/to/rbmd
  RBMD_JSON=/absolute/path/to/h2o.json

Optional:
  MPI_RANKS=1                 MPI ranks to launch
  RBMD_WORK_DIR=<json-dir>    Working directory for relative input paths
  DIAG_DIR=<cwd>/diagnostics/cuda_memcheck_<timestamp>
  SANITIZER_PADDING=32        Bytes placed after every supported allocation
  SANITIZER_ERROR_EXITCODE=86 Exit code used when memcheck detects an error
  MPI_EXTRA_ARGS=""           Extra whitespace-separated mpirun arguments

Recommended CUDA diagnostic build:
  cmake -S . -B build-cuda-memcheck \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_FLAGS=-lineinfo \
    -DUSE_CUDA=ON -DUSE_ROCM=OFF -DUSE_MPI=ON -DWARP_SIZE_64=OFF
  cmake --build build-cuda-memcheck -j

Examples:
  MPI_RANKS=1 RBMD_BIN=./build-cuda-memcheck/rbmd \
    RBMD_JSON=/path/to/h2o.json scripts/run_cuda_memcheck_mpi.sh

  MPI_RANKS=4 RBMD_BIN=./build-cuda-memcheck/rbmd \
    RBMD_JSON=/path/to/h2o.json scripts/run_cuda_memcheck_mpi.sh
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got: ${value}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
[[ "$#" -eq 0 ]] || die "unexpected arguments; use environment variables or --help"

require_command compute-sanitizer
require_command mpirun
require_command realpath
require_command tee

[[ -n "${RBMD_BIN:-}" ]] || die "RBMD_BIN is required"
[[ -n "${RBMD_JSON:-}" ]] || die "RBMD_JSON is required"

RBMD_BIN="$(realpath "${RBMD_BIN}")"
RBMD_JSON="$(realpath "${RBMD_JSON}")"
[[ -x "${RBMD_BIN}" ]] || die "RBMD_BIN is not executable: ${RBMD_BIN}"
[[ -r "${RBMD_JSON}" ]] || die "RBMD_JSON is not readable: ${RBMD_JSON}"

MPI_RANKS="${MPI_RANKS:-1}"
SANITIZER_PADDING="${SANITIZER_PADDING:-32}"
SANITIZER_ERROR_EXITCODE="${SANITIZER_ERROR_EXITCODE:-86}"
require_nonnegative_integer MPI_RANKS "${MPI_RANKS}"
require_nonnegative_integer SANITIZER_PADDING "${SANITIZER_PADDING}"
require_nonnegative_integer SANITIZER_ERROR_EXITCODE "${SANITIZER_ERROR_EXITCODE}"
[[ "${MPI_RANKS}" -gt 0 ]] || die "MPI_RANKS must be greater than zero"

RBMD_WORK_DIR="${RBMD_WORK_DIR:-$(dirname "${RBMD_JSON}")}"
RBMD_WORK_DIR="$(realpath "${RBMD_WORK_DIR}")"
[[ -d "${RBMD_WORK_DIR}" ]] || die "RBMD_WORK_DIR is not a directory: ${RBMD_WORK_DIR}"

DIAG_DIR="${DIAG_DIR:-$(pwd)/diagnostics/cuda_memcheck_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${DIAG_DIR}"
DIAG_DIR="$(realpath "${DIAG_DIR}")"

read -r -a MPI_EXTRA_ARGS_ARRAY <<< "${MPI_EXTRA_ARGS:-}"

SANITIZER_LOG_PATTERN="${DIAG_DIR}/memcheck.rank-%q{OMPI_COMM_WORLD_RANK}.pid-%p.log"
SANITIZER_ARGS=(
  --tool memcheck
  --padding "${SANITIZER_PADDING}"
  --error-exitcode "${SANITIZER_ERROR_EXITCODE}"
  --launch-timeout 0
  --report-api-errors no
  --show-backtrace yes
  --log-file "${SANITIZER_LOG_PATTERN}"
)

MPI_COMMAND=(
  mpirun
  -n "${MPI_RANKS}"
  "${MPI_EXTRA_ARGS_ARRAY[@]}"
  compute-sanitizer
  "${SANITIZER_ARGS[@]}"
  --
  "${RBMD_BIN}"
  -j "${RBMD_JSON}"
)

{
  printf 'timestamp=%s\n' "$(date '+%F %T %z')"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'work_dir=%s\n' "${RBMD_WORK_DIR}"
  printf 'rbmd_bin=%s\n' "${RBMD_BIN}"
  printf 'rbmd_json=%s\n' "${RBMD_JSON}"
  printf 'mpi_ranks=%s\n' "${MPI_RANKS}"
  printf 'sanitizer_padding=%s\n' "${SANITIZER_PADDING}"
  printf 'sanitizer_error_exitcode=%s\n' "${SANITIZER_ERROR_EXITCODE}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
  printf 'RBMD_GPU_LIST=%s\n' "${RBMD_GPU_LIST:-<unset>}"
  printf 'RBMD_GPU_BINDING_STRICT=%s\n' "${RBMD_GPU_BINDING_STRICT:-<unset>}"
  printf 'compute_sanitizer_version='
  compute-sanitizer --version 2>&1 | tr '\n' ' '
  printf '\ncommand='
  printf '%q ' "${MPI_COMMAND[@]}"
  printf '\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf '\n[nvidia-smi -L]\n'
    nvidia-smi -L || true
  fi
} > "${DIAG_DIR}/run_info.log" 2>&1

export RBMD_GPU_BINDING_VERBOSE="${RBMD_GPU_BINDING_VERBOSE:-1}"

printf 'Compute Sanitizer reports: %s\n' "${SANITIZER_LOG_PATTERN}"
printf 'Launcher output: %s\n' "${DIAG_DIR}/launcher.log"
printf 'Command: '
printf '%q ' "${MPI_COMMAND[@]}"
printf '\n'

cd "${RBMD_WORK_DIR}"
set +e
"${MPI_COMMAND[@]}" 2>&1 | tee "${DIAG_DIR}/launcher.log"
mpi_status="${PIPESTATUS[0]}"
set -e

SUMMARY_PATTERN='Invalid __|out of bounds|misaligned|Hardware exception|Program hit|CUDA API error|ERROR SUMMARY: [1-9]|Target application returned an error|Sanitizer encountered an error'
{
  printf 'mpi_exit_code=%s\n' "${mpi_status}"
  printf 'generated_at=%s\n\n' "$(date '+%F %T %z')"
  grep -HEn "${SUMMARY_PATTERN}" "${DIAG_DIR}"/memcheck.*.log "${DIAG_DIR}/launcher.log" 2>/dev/null || true
} > "${DIAG_DIR}/summary.log"

printf 'MPI exit code: %s\n' "${mpi_status}"
printf 'Summary: %s\n' "${DIAG_DIR}/summary.log"
exit "${mpi_status}"
