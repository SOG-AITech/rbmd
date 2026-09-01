#!/bin/bash
# Diagnostic Slurm script for Kunshan runh2o.
# It keeps the same resource shape as flexible.slurm and records CPU memory,
# DCU/ROCm memory, process state, filesystem state, and job accounting.

#SBATCH -J rbmd-dtk-diag
#SBATCH -p kshdnormal02
#SBATCH -N 4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --output=100M_diag_%j.out

set -uo pipefail

module purge && module load compiler/devtoolset/7.3.1 mpi/hpcx/gcc-7.3.1 compiler/dtk/24.04 compiler/cmake/3.23.3

export ROCM_PATH=/public/software/compiler/rocm/dtk-24.04
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH}

ulimit -c 0

RUN_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RBMD_JSON="${RBMD_JSON:-./h2o.json}"
RBMD_DIAG_INTERVAL="${RBMD_DIAG_INTERVAL:-10}"
JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
DIAG_DIR="${RUN_DIR}/diagnostics/${JOB_TAG}"
mkdir -p "${DIAG_DIR}"

print_header() {
  local title="$1"
  printf '\n========== %s ==========\n' "${title}"
}

snapshot_node() {
  local tag="$1"
  local out="${DIAG_DIR}/${tag}_${HOSTNAME}.log"
  {
    print_header "snapshot ${tag}"
    date '+%F %T %z'
    echo "hostname=${HOSTNAME}"
    echo "pwd=$(pwd)"

    print_header "uptime"
    uptime || true

    print_header "free"
    free -h || true

    print_header "proc meminfo"
    grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|CommitLimit|Committed_AS|HugePages' /proc/meminfo || true

    print_header "df"
    df -h "${RUN_DIR}" || true

    print_header "rocm-smi"
    rocm-smi || true

    print_header "rocm-smi vram"
    rocm-smi --showmeminfo vram || true

    print_header "rocm-smi pid"
    rocm-smi --showpidgpus || rocm-smi --showpids || true

    print_header "rbmd processes"
    ps -u "${USER}" -o pid,ppid,stat,etime,rss,vsz,pcpu,pmem,comm,args \
      | awk '/rbmd|mpirun|orted|prterun|srun/ && !/awk/ {print} END {exit 0}'
  } >> "${out}" 2>&1
}

node_monitor_loop() {
  local interval="$1"
  local out="${DIAG_DIR}/node_monitor_${HOSTNAME}.log"
  while true; do
    {
      print_header "sample"
      date '+%F %T %z'
      echo "hostname=${HOSTNAME}"

      print_header "free"
      free -h || true

      print_header "rocm-smi vram"
      rocm-smi --showmeminfo vram || true

      print_header "rocm-smi pid"
      rocm-smi --showpidgpus || rocm-smi --showpids || true

      print_header "rbmd processes"
      ps -u "${USER}" -o pid,ppid,stat,etime,rss,vsz,pcpu,pmem,comm,args \
        | awk '/rbmd|mpirun|orted|prterun|srun/ && !/awk/ {print} END {exit 0}'
    } >> "${out}" 2>&1
    sleep "${interval}"
  done
}

write_job_info() {
  {
    print_header "job"
    date '+%F %T %z'
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
    echo "SLURM_NNODES=${SLURM_NNODES:-}"
    echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
    echo "SLURM_TASKS_PER_NODE=${SLURM_TASKS_PER_NODE:-}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
    echo "RBMD_JSON=${RBMD_JSON}"
    echo "RBMD_DIAG_INTERVAL=${RBMD_DIAG_INTERVAL}"
    echo "RUN_DIR=${RUN_DIR}"
    echo "DIAG_DIR=${DIAG_DIR}"

    print_header "nodes"
    scontrol show hostnames "${SLURM_JOB_NODELIST:-}" || true

    print_header "scontrol"
    scontrol show job "${SLURM_JOB_ID:-}" || true

    print_header "environment"
    env | sort

    print_header "input json"
    sed -n '1,240p' "${RBMD_JSON}" || true

    print_header "binary"
    ls -lah ../rbmd || true
    ldd ../rbmd || true
  } > "${DIAG_DIR}/job_info.log" 2>&1
}

write_final_info() {
  local exit_code="$1"
  {
    print_header "final"
    date '+%F %T %z'
    echo "rbmd_exit_code=${exit_code}"

    print_header "output files"
    ls -lah "${RUN_DIR}" || true
    stat -c '%y %s %n' "${RUN_DIR}"/100M*.out "${RUN_DIR}"/rbmd.log \
      "${RUN_DIR}"/temperature.txt "${RUN_DIR}"/thermo.txt "${RUN_DIR}"/rbmd.trj 2>/dev/null || true

    print_header "tail 100M outputs"
    tail -120 "${RUN_DIR}"/100M*.out 2>/dev/null || true

    print_header "tail rbmd.log"
    tail -160 "${RUN_DIR}/rbmd.log" 2>/dev/null || true

    print_header "sacct"
    sacct -j "${SLURM_JOB_ID:-}" \
      --format=JobID,JobName%30,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AveRSS,MaxVMSize,ReqMem,NNodes,NTasks,NodeList%80 \
      -P 2>/dev/null || true
  } > "${DIAG_DIR}/final_info.log" 2>&1
}

cleanup() {
  local exit_code="$?"
  if [[ -n "${LOCAL_MONITOR_PID:-}" ]]; then
    kill "${LOCAL_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SRUN_MONITOR_PID:-}" ]]; then
    kill "${SRUN_MONITOR_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  write_final_info "${exit_code}"
  exit "${exit_code}"
}
trap cleanup EXIT

cd "${RUN_DIR}" || exit 1

write_job_info
snapshot_node "batch_start"

# Batch-node monitor. This always works and is useful even if the per-node srun
# monitor is blocked by site Slurm settings.
node_monitor_loop "${RBMD_DIAG_INTERVAL}" &
LOCAL_MONITOR_PID="$!"

# One monitor task per allocated node. --overlap is required on many Slurm sites
# when a monitor step runs concurrently with mpirun. If unsupported, the command
# exits and the batch-node monitor still remains active.
srun --overlap --ntasks="${SLURM_NNODES:-1}" --ntasks-per-node=1 \
  bash -lc "$(declare -f print_header node_monitor_loop); DIAG_DIR='${DIAG_DIR}'; USER='${USER}'; node_monitor_loop '${RBMD_DIAG_INTERVAL}'" \
  > "${DIAG_DIR}/srun_monitor_launcher.log" 2>&1 &
SRUN_MONITOR_PID="$!"

sleep 3

print_header "starting rbmd"
date '+%F %T %z'
mpirun -n 16 --mca pml ucx -x UCX_TLS=^cma \
  -x LD_LIBRARY_PATH \
  -x ROCM_PATH \
  ../rbmd -j "${RBMD_JSON}"
RBMD_EXIT_CODE="$?"

snapshot_node "batch_after_rbmd"
exit "${RBMD_EXIT_CODE}"
