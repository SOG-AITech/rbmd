#!/usr/bin/env bash
set -euo pipefail

# Build UCX and Open MPI from GitHub, then generate an environment file for rbmd.
#
# Common usage:
#   scripts/install_mpi_runtime.sh
#   source ../depend/mpi_env.sh
#
# Useful overrides:
#   PACKAGE_ROOT=/path/to/rbmd-package CUDA_HOME=/usr/local/cuda scripts/install_mpi_runtime.sh
#   PREFIX_ROOT=/path/to/rbmd-package/depend scripts/install_mpi_runtime.sh
#   UCX_REF=v1.18.1 OMPI_REF=v5.0.8 scripts/install_mpi_runtime.sh
#   WITH_CUDA=OFF scripts/install_mpi_runtime.sh
#   INSTALL_SHELL_ENV=ON scripts/install_mpi_runtime.sh
#   SHELL_RC_FILE="$HOME/.profile" scripts/install_mpi_runtime.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

PREFIX_ROOT="${PREFIX_ROOT:-${PACKAGE_ROOT}/depend}"
UCX_PREFIX="${UCX_PREFIX:-${PREFIX_ROOT}/UCX}"
OMPI_PREFIX="${OMPI_PREFIX:-${PREFIX_ROOT}/OpenMPI}"
ENV_FILE="${ENV_FILE:-${PREFIX_ROOT}/mpi_env.sh}"

WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/mpi-runtime-work}"
SRC_ROOT="${SRC_ROOT:-${WORK_ROOT}/src}"
BUILD_DIR="${BUILD_DIR:-${WORK_ROOT}/build}"

UCX_REPO="${UCX_REPO:-https://github.com/openucx/ucx.git}"
OMPI_REPO="${OMPI_REPO:-https://github.com/open-mpi/ompi.git}"
UCX_REF="${UCX_REF:-latest}"
OMPI_REF="${OMPI_REF:-latest}"

CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
WITH_CUDA="${WITH_CUDA:-ON}"
JOBS="${JOBS:-$(nproc)}"
REBUILD="${REBUILD:-OFF}"
OMPI_DISABLE_BTL_UCT="${OMPI_DISABLE_BTL_UCT:-ON}"
INSTALL_SHELL_ENV="${INSTALL_SHELL_ENV:-ASK}"
SHELL_RC_FILE="${SHELL_RC_FILE:-${HOME}/.bashrc}"

UCX_CONFIGURE_ARGS="${UCX_CONFIGURE_ARGS:-}"
OMPI_CONFIGURE_ARGS="${OMPI_CONFIGURE_ARGS:-}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

detect_latest_tag() {
  local repo="$1"
  local tag
  tag="$(
    git ls-remote --tags --refs "$repo" 'v[0-9]*' |
      awk '{ sub("refs/tags/", "", $2); print $2 }' |
      grep -Ev '(rc|alpha|beta|pre)' |
      sort -V |
      tail -n 1
  )"
  [[ -n "$tag" ]] || die "failed to detect latest release tag from $repo"
  printf '%s\n' "$tag"
}

checkout_repo() {
  local name="$1"
  local repo="$2"
  local ref="$3"
  local dst="${SRC_ROOT}/${name}"

  mkdir -p "$SRC_ROOT"
  if [[ -d "${dst}/.git" ]]; then
    log "Updating ${name}: ${dst}"
    git -C "$dst" fetch --tags --prune origin >&2
  else
    log "Cloning ${name}: ${repo}"
    git clone --recursive "$repo" "$dst" >&2
  fi

  if [[ "$ref" == "latest" ]]; then
    ref="$(detect_latest_tag "$repo")"
  fi

  log "Checking out ${name}: ${ref}"
  git -C "$dst" checkout --detach "$ref" >&2
  git -C "$dst" submodule update --init --recursive >&2
  printf '%s\n' "$dst"
}

ensure_configure() {
  local name="$1"
  local src="$2"

  if [[ -x "${src}/configure" ]]; then
    return
  fi

  log "Generating configure script for ${name}"
  case "$name" in
    ucx)
      [[ -x "${src}/autogen.sh" ]] || die "UCX configure missing and autogen.sh not found"
      (cd "$src" && ./autogen.sh)
      ;;
    openmpi)
      [[ -x "${src}/autogen.pl" ]] || die "Open MPI configure missing and autogen.pl not found"
      (cd "$src" && ./autogen.pl)
      ;;
    *)
      die "unknown project: $name"
      ;;
  esac

  [[ -x "${src}/configure" ]] || die "${name} did not generate configure"
}

configure_cuda_args() {
  if [[ "$WITH_CUDA" != "ON" ]]; then
    return
  fi
  [[ -d "$CUDA_HOME" ]] || die "CUDA_HOME does not exist: $CUDA_HOME"
  printf -- '--with-cuda=%s\n' "$CUDA_HOME"
}

build_ucx() {
  local src="$1"
  local build="${BUILD_DIR}/ucx"
  local cuda_arg
  cuda_arg="$(configure_cuda_args)"

  ensure_configure ucx "$src"

  if [[ "$REBUILD" == "ON" ]]; then
    rm -rf "$build"
  fi
  mkdir -p "$build"

  log "Configuring UCX"
  (
    cd "$build"
    "${src}/configure" \
      --prefix="$UCX_PREFIX" \
      ${cuda_arg:+"$cuda_arg"} \
      $UCX_CONFIGURE_ARGS
  )

  log "Building UCX"
  make -C "$build" -j"$JOBS"
  make -C "$build" install
}

build_openmpi() {
  local src="$1"
  local build="${BUILD_DIR}/openmpi"
  local cuda_arg
  local btl_uct_arg=()
  cuda_arg="$(configure_cuda_args)"

  ensure_configure openmpi "$src"

  if [[ "$OMPI_DISABLE_BTL_UCT" == "ON" ]]; then
    btl_uct_arg=(--enable-mca-no-build=btl-uct)
  fi

  if [[ "$REBUILD" == "ON" ]]; then
    rm -rf "$build"
  fi
  mkdir -p "$build"

  log "Configuring Open MPI"
  (
    cd "$build"
    "${src}/configure" \
      --prefix="$OMPI_PREFIX" \
      --with-ucx="$UCX_PREFIX" \
      ${cuda_arg:+"$cuda_arg"} \
      "${btl_uct_arg[@]}" \
      $OMPI_CONFIGURE_ARGS
  )

  log "Building Open MPI"
  make -C "$build" -j"$JOBS"
  make -C "$build" install
}

write_env_file() {
  mkdir -p "$(dirname "$ENV_FILE")"
  cat >"$ENV_FILE" <<EOF
# Source this file before building or running rbmd with this MPI runtime:
#   source "$ENV_FILE"

export UCX_HOME="$UCX_PREFIX"
export MPI_HOME="$OMPI_PREFIX"
export OPAL_PREFIX="$OMPI_PREFIX"
export CUDA_HOME="$CUDA_HOME"

_rbmd_prepend_path() {
  local var="\$1"
  local value="\$2"
  local current="\${!var:-}"
  [[ -d "\$value" ]] || return 0
  case ":\$current:" in
    *:"\$value":*) ;;
    *) export "\$var=\$value\${current:+:\$current}" ;;
  esac
}

_rbmd_prepend_path PATH "\$MPI_HOME/bin"
_rbmd_prepend_path LD_LIBRARY_PATH "\$MPI_HOME/lib"
_rbmd_prepend_path LD_LIBRARY_PATH "\$MPI_HOME/lib64"
_rbmd_prepend_path LD_LIBRARY_PATH "\$UCX_HOME/lib"
_rbmd_prepend_path LD_LIBRARY_PATH "\$UCX_HOME/lib64"
_rbmd_prepend_path LD_LIBRARY_PATH "\$CUDA_HOME/lib64"
_rbmd_prepend_path LD_LIBRARY_PATH "\$CUDA_HOME/lib"
_rbmd_prepend_path LIBRARY_PATH "\$MPI_HOME/lib"
_rbmd_prepend_path LIBRARY_PATH "\$MPI_HOME/lib64"
_rbmd_prepend_path LIBRARY_PATH "\$UCX_HOME/lib"
_rbmd_prepend_path LIBRARY_PATH "\$UCX_HOME/lib64"
_rbmd_prepend_path CPATH "\$MPI_HOME/include"
_rbmd_prepend_path CPATH "\$UCX_HOME/include"
_rbmd_prepend_path PKG_CONFIG_PATH "\$MPI_HOME/lib/pkgconfig"
_rbmd_prepend_path PKG_CONFIG_PATH "\$MPI_HOME/lib64/pkgconfig"
_rbmd_prepend_path PKG_CONFIG_PATH "\$UCX_HOME/lib/pkgconfig"
_rbmd_prepend_path PKG_CONFIG_PATH "\$UCX_HOME/lib64/pkgconfig"
_rbmd_prepend_path CMAKE_PREFIX_PATH "\$MPI_HOME"
_rbmd_prepend_path CMAKE_PREFIX_PATH "\$UCX_HOME"
_rbmd_prepend_path MANPATH "\$MPI_HOME/share/man"

# Prefer the UCX path for CUDA-aware Open MPI runs. Override these before source
# if a specific cluster requires different MCA settings.
export OMPI_MCA_pml="\${OMPI_MCA_pml:-ucx}"
export OMPI_MCA_osc="\${OMPI_MCA_osc:-ucx}"
export OMPI_MCA_btl="\${OMPI_MCA_btl:-^uct,openib}"

unset -f _rbmd_prepend_path
EOF
}

detect_shell_rc_file() {
  printf '%s\n' "$SHELL_RC_FILE"
}

write_shell_env_hook() {
  local rc_file="$1"
  local marker_begin="# >>> rbmd mpi runtime >>>"
  local marker_end="# <<< rbmd mpi runtime <<<"

  mkdir -p "$(dirname "$rc_file")"
  touch "$rc_file"

  if grep -Fq "$marker_begin" "$rc_file"; then
    log "Shell environment hook already exists: ${rc_file}"
    return
  fi

  cat >>"$rc_file" <<EOF

${marker_begin}
if [ -f "$ENV_FILE" ]; then
  . "$ENV_FILE"
fi
${marker_end}
EOF

  log "Wrote shell environment hook: ${rc_file}"
}

maybe_install_shell_env() {
  local rc_file
  rc_file="$(detect_shell_rc_file)"

  case "$INSTALL_SHELL_ENV" in
    ON|on|YES|yes|1|true|TRUE)
      write_shell_env_hook "$rc_file"
      return
      ;;
    OFF|off|NO|no|0|false|FALSE)
      log "Skipped shell environment hook"
      return
      ;;
    ASK|ask)
      ;;
    *)
      die "INSTALL_SHELL_ENV must be ASK, ON, or OFF, got: $INSTALL_SHELL_ENV"
      ;;
  esac

  if [[ ! -t 0 ]]; then
    log "Non-interactive shell detected; skipped shell environment hook"
    printf 'To enable manually, run:\n'
    printf '  source "%s"\n' "$ENV_FILE"
    return
  fi

  printf '\n是否写入当前用户环境变量启动文件？\n'
  printf '目标文件：%s\n' "$rc_file"
  printf '写入后新终端会自动加载 MPI/UCX 环境；当前终端仍需 source 一次。\n'
  read -r -p '写入吗？[y/N] ' answer
  case "$answer" in
    y|Y|yes|YES)
      write_shell_env_hook "$rc_file"
      ;;
    *)
      log "Skipped shell environment hook"
      printf 'Manual command:\n'
      printf '  source "%s"\n' "$ENV_FILE"
      ;;
  esac
}

verify_install() {
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  log "MPI runtime summary"
  command -v mpirun
  mpirun --version | head -n 1
  mpicc --showme:link || true
  ompi_info --parsable --all | grep -E '(^mca:pml:ucx|mpi_built_with_cuda_support|opal_cuda_support)' || true
}

main() {
  need_cmd git
  need_cmd make
  need_cmd awk
  need_cmd sort
  need_cmd grep
  need_cmd head
  need_cmd tail
  need_cmd nproc

  log "Install prefixes"
  printf 'UCX_PREFIX=%s\n' "$UCX_PREFIX"
  printf 'OMPI_PREFIX=%s\n' "$OMPI_PREFIX"
  printf 'ENV_FILE=%s\n' "$ENV_FILE"
  printf 'PACKAGE_ROOT=%s\n' "$PACKAGE_ROOT"
  printf 'WORK_ROOT=%s\n' "$WORK_ROOT"
  printf 'CUDA_HOME=%s\n' "$CUDA_HOME"

  local ucx_src
  local ompi_src
  ucx_src="$(checkout_repo ucx "$UCX_REPO" "$UCX_REF")"
  ompi_src="$(checkout_repo openmpi "$OMPI_REPO" "$OMPI_REF")"

  build_ucx "$ucx_src"
  build_openmpi "$ompi_src"
  write_env_file
  maybe_install_shell_env
  verify_install

  log "Done"
  printf 'Run this before building/running rbmd:\n'
  printf '  source "%s"\n' "$ENV_FILE"
}

main "$@"
