#!/usr/bin/env bash
set -euo pipefail

# Install a relocatable Python runtime for RBMD analysis postprocess.
#
# Common usage inside an unpacked release package:
#   ./scripts/install_python_runtime.sh
#   ./python/bin/python -c "import numpy, freud"
#
# Useful overrides:
#   PACKAGE_ROOT=/path/to/rbmd-release ./scripts/install_python_runtime.sh
#   PYTHON_VERSION=3.10 ./scripts/install_python_runtime.sh
#   PYTHON_RUNTIME_URL=https://.../cpython-...-install_only_stripped.tar.gz ./scripts/install_python_runtime.sh
#   PYTHON_PACKAGES="numpy freud-analysis" ./scripts/install_python_runtime.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_PREFIX="${PYTHON_PREFIX:-${PACKAGE_ROOT}/python}"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/python-runtime-work}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${WORK_ROOT}/downloads}"
EXTRACT_DIR="${EXTRACT_DIR:-${WORK_ROOT}/extract}"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTHON_PACKAGES="${PYTHON_PACKAGES:-numpy freud-analysis}"
PYTHON_BUILD_STANDALONE_REPO="${PYTHON_BUILD_STANDALONE_REPO:-astral-sh/python-build-standalone}"
PYTHON_RUNTIME_URL="${PYTHON_RUNTIME_URL:-}"

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

detect_target_triple() {
  local machine
  machine="$(uname -m)"
  case "$machine" in
    x86_64|amd64)
      printf '%s\n' "x86_64-unknown-linux-gnu"
      ;;
    aarch64|arm64)
      printf '%s\n' "aarch64-unknown-linux-gnu"
      ;;
    *)
      die "unsupported architecture: ${machine}. Set PYTHON_RUNTIME_URL manually."
      ;;
  esac
}

select_asset_url() {
  local api_url="$1"
  local target_triple="$2"
  local version_prefix="$3"

  python3 - "$api_url" "$target_triple" "$version_prefix" <<'PY'
import json
import sys
import urllib.request

api_url, target_triple, version_prefix = sys.argv[1:4]
with urllib.request.urlopen(api_url, timeout=60) as response:
    release = json.load(response)

assets = release.get("assets", [])
matches = []
for asset in assets:
    name = asset.get("name", "")
    url = asset.get("browser_download_url", "")
    if not url:
        continue
    if not name.startswith(f"cpython-{version_prefix}."):
        continue
    if target_triple not in name:
        continue
    if "install_only" not in name:
        continue
    if not (name.endswith(".tar.gz") or name.endswith(".tar.zst")):
        continue
    matches.append((name, url))

if not matches:
    names = "\n".join(asset.get("name", "") for asset in assets)
    raise SystemExit(
        "no matching python-build-standalone asset found for "
        f"Python {version_prefix}, {target_triple}\n"
        "Set PYTHON_RUNTIME_URL manually.\n"
        "Available assets:\n"
        f"{names}"
    )

def score(item):
    name, _ = item
    return (
        1 if "install_only_stripped" in name else 0,
        1 if name.endswith(".tar.zst") else 0,
        name,
    )

print(sorted(matches, key=score, reverse=True)[0][1])
PY
}

download_runtime() {
  local url="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  log "Downloading Python runtime"
  printf 'URL=%s\n' "$url"
  curl -fL --retry 3 --retry-delay 2 -o "$dst" "$url"
}

extract_runtime() {
  local archive="$1"
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"

  log "Extracting Python runtime"
  case "$archive" in
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$EXTRACT_DIR"
      ;;
    *.tar.zst)
      need_cmd zstd
      tar --use-compress-program zstd -xf "$archive" -C "$EXTRACT_DIR"
      ;;
    *)
      die "unsupported archive format: $archive"
      ;;
  esac
}

find_python_bin_name() {
  local root="$1"
  local exe
  for exe in \
    "${root}/bin/python" \
    "${root}/bin/python3" \
    "${root}/bin/python3.10" \
    "${root}/bin/python3."*; do
    if [[ -x "$exe" ]]; then
      basename "$exe"
      return
    fi
  done
}

find_extracted_python_root() {
  local candidate
  while IFS= read -r candidate; do
    if [[ -n "$(find_python_bin_name "$candidate")" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done < <(find "$EXTRACT_DIR" -type d \( -name python -o -name install \) -print)

  die "extracted archive does not contain bin/python"
}

install_runtime_tree() {
  local src_root="$1"
  log "Installing Python runtime to ${PYTHON_PREFIX}"
  rm -rf "$PYTHON_PREFIX"
  mkdir -p "$PYTHON_PREFIX"
  cp -a "${src_root}/." "$PYTHON_PREFIX/"

  local python_bin_name
  python_bin_name="$(find_python_bin_name "$PYTHON_PREFIX")"
  [[ -n "$python_bin_name" ]] || die "missing Python executable under ${PYTHON_PREFIX}/bin"

  if [[ "$python_bin_name" != "python" && ! -e "${PYTHON_PREFIX}/bin/python" ]]; then
    ln -s "$python_bin_name" "${PYTHON_PREFIX}/bin/python"
  fi

  [[ -x "${PYTHON_PREFIX}/bin/python" ]] || die "missing ${PYTHON_PREFIX}/bin/python"
  [[ ! -f "${PYTHON_PREFIX}/pyvenv.cfg" ]] ||
    die "unexpected venv metadata found: ${PYTHON_PREFIX}/pyvenv.cfg"
}

install_packages() {
  log "Installing analysis Python packages"
  "${PYTHON_PREFIX}/bin/python" -m ensurepip --upgrade || true
  "${PYTHON_PREFIX}/bin/python" -m pip install --upgrade pip
  # shellcheck disable=SC2086
  "${PYTHON_PREFIX}/bin/python" -m pip install ${PYTHON_PACKAGES}
}

verify_install() {
  log "Verifying Python runtime"
  [[ ! -f "${PYTHON_PREFIX}/pyvenv.cfg" ]] ||
    die "do not package venv metadata: ${PYTHON_PREFIX}/pyvenv.cfg"

  "${PYTHON_PREFIX}/bin/python" - <<'PY'
import sys
import numpy
import freud

print(sys.executable)
print("numpy", numpy.__version__)
print("freud", freud.__version__)
PY
}

main() {
  need_cmd curl
  need_cmd tar
  need_cmd python3

  mkdir -p "$DOWNLOAD_DIR"

  local target_triple
  target_triple="$(detect_target_triple)"

  local runtime_url
  runtime_url="$PYTHON_RUNTIME_URL"
  if [[ -z "$runtime_url" ]]; then
    runtime_url="$(select_asset_url \
      "https://api.github.com/repos/${PYTHON_BUILD_STANDALONE_REPO}/releases/latest" \
      "$target_triple" \
      "$PYTHON_VERSION")"
  fi

  local archive_name
  archive_name="${runtime_url##*/}"
  local archive_path="${DOWNLOAD_DIR}/${archive_name}"

  log "Install prefixes"
  printf 'PACKAGE_ROOT=%s\n' "$PACKAGE_ROOT"
  printf 'PYTHON_PREFIX=%s\n' "$PYTHON_PREFIX"
  printf 'PYTHON_VERSION=%s\n' "$PYTHON_VERSION"
  printf 'TARGET_TRIPLE=%s\n' "$target_triple"
  printf 'PYTHON_PACKAGES=%s\n' "$PYTHON_PACKAGES"

  download_runtime "$runtime_url" "$archive_path"
  extract_runtime "$archive_path"

  local src_root
  src_root="$(find_extracted_python_root)"
  install_runtime_tree "$src_root"
  install_packages
  verify_install

  log "Done"
  printf 'Python runtime is ready:\n'
  printf '  %s/bin/python\n' "$PYTHON_PREFIX"
}

main "$@"
