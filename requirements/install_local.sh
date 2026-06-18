#!/bin/bash
# install_local.sh - Run RLinf install.sh with all downloads redirected to the current working directory.
#
# Usage (from repository root):
#   bash requirements/install_local.sh embodied --model openpi --env maniskill_libero --use-mirror
#
# This wrapper ensures that:
#   - The virtual environment is created under ./.venv
#   - uv package cache goes to ./.uv_cache
#   - HuggingFace cache goes to ./.hf_home
#   - LIBERO is cloned to ./libero
#   - ManiSkill assets go to ./.maniskill
#   - SAPIEN PhysX libraries go to ./.sapien
#   - OpenPI tokenizer goes to ./.cache/openpi
#
# Note: requirements/embodied/sys_deps.sh may still install system packages
# to /usr/lib etc. Run with --no-root if you want to skip system deps (make
# sure they are already installed).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repository root by walking up until we find requirements/install.sh.
REPO_DIR="$SCRIPT_DIR"
while [[ ! -f "$REPO_DIR/requirements/install.sh" && "$REPO_DIR" != "/" ]]; do
    REPO_DIR="$(cd "$REPO_DIR/.." && pwd)"
done

if [[ ! -f "$REPO_DIR/requirements/install.sh" ]]; then
    echo "[install_local.sh] Cannot locate requirements/install.sh." >&2
    echo "Please run this script from inside the RLinf repository." >&2
    exit 1
fi

WORK_DIR="$(pwd)"

if [[ "$WORK_DIR" != "$REPO_DIR" ]]; then
    echo "[install_local.sh] Running outside of repository root is not recommended."
    echo "  Repository root: $REPO_DIR"
    echo "  Current directory: $WORK_DIR"
    echo "Please cd to the repository root and try again."
    exit 1
fi

# Parse our own flags first. Anything else is passed through to install.sh.
FORCE=0
INSTALL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        *)
            INSTALL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Create cache directories upfront. Do NOT create asset directories like
# .maniskill/ here: install.sh's download_assets.sh uses the *existence* of
# those directories as a skip signal, so an empty directory would incorrectly
# prevent assets from being downloaded.
mkdir -p "$WORK_DIR/.uv_cache"
mkdir -p "$WORK_DIR/.hf_home/hub"
mkdir -p "$WORK_DIR/.cache/openpi"

# Redirect all user-level downloads to the current directory.
export DOWNLOAD_DIR="$WORK_DIR"
export LIBERO_PATH="$WORK_DIR/libero"
export UV_CACHE_DIR="$WORK_DIR/.uv_cache"
export HF_HOME="$WORK_DIR/.hf_home"
export HF_HUB_CACHE="$WORK_DIR/.hf_home/hub"
export OPENPI_DATA_HOME="$WORK_DIR/.cache/openpi"

# Optional: enable hf-transfer for faster HuggingFace downloads.
# If hf-transfer is not installed, huggingface_hub will fall back to the default downloader.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Force the virtual environment to be created in the current directory.
INSTALL_ARGS+=("--venv" "$WORK_DIR/.venv")

echo "[install_local.sh] ============================================================"
echo "[install_local.sh] Running RLinf installer with local download paths:"
echo "[install_local.sh]   DOWNLOAD_DIR      = $DOWNLOAD_DIR"
echo "[install_local.sh]   LIBERO_PATH       = $LIBERO_PATH"
echo "[install_local.sh]   UV_CACHE_DIR      = $UV_CACHE_DIR"
echo "[install_local.sh]   HF_HOME           = $HF_HOME"
echo "[install_local.sh]   HF_HUB_CACHE      = $HF_HUB_CACHE"
echo "[install_local.sh]   OPENPI_DATA_HOME  = $OPENPI_DATA_HOME"
echo "[install_local.sh]   VENV_DIR          = $WORK_DIR/.venv"
echo "[install_local.sh] ============================================================"

# Lightweight idempotency check: if the previous install looks complete and the
# user did not request --force, skip the full install.sh flow. This avoids the
# repeated uv dependency reconciliation (uninstall/reinstall dance) that
# install.sh performs on every run.

# Return 0 if the directory exists and contains at least one entry.
dir_has_content() {
    local d="$1"
    [[ -d "$d" ]] || return 1
    local count
    count=$(find "$d" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | wc -l)
    [[ "$count" -gt 0 ]]
}

looks_complete() {
    [[ "$FORCE" == "1" ]] && return 1
    [[ -f "$WORK_DIR/.venv/bin/python" ]] || return 1
    local active_mm
    active_mm=$("$WORK_DIR/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
    [[ "$active_mm" == "3.11" ]] || return 1
    # LIBERO must actually be cloned (not just an empty directory).
    [[ -f "$WORK_DIR/libero/setup.py" || -f "$WORK_DIR/libero/pyproject.toml" ]] || return 1
    # ManiSkill assets must be non-empty.
    dir_has_content "$WORK_DIR/.maniskill" || return 1
    # SAPIEN PhysX libraries must be non-empty.
    dir_has_content "$WORK_DIR/.sapien/physx/105.1-physx-5.3.1.patch0" || return 1
    # OpenPI tokenizer marker. Note: hf download puts it under big_vision/.
    [[ -f "$WORK_DIR/.cache/openpi/big_vision/paligemma_tokenizer.model" ]] || return 1
    return 0
}

if looks_complete; then
    echo "[install_local.sh] Existing installation looks complete. Skipping full install.sh to avoid"
    echo "[install_local.sh] repeated dependency reconciliation. Use --force to run the full install anyway."
    echo "[install_local.sh] Installation finished. Local assets located at:"
    echo "[install_local.sh]   Virtual env:     $WORK_DIR/.venv"
    echo "[install_local.sh]   uv cache:        $WORK_DIR/.uv_cache"
    echo "[install_local.sh]   HF cache:        $WORK_DIR/.hf_home"
    echo "[install_local.sh]   LIBERO:          $WORK_DIR/libero"
    echo "[install_local.sh]   ManiSkill:       $WORK_DIR/.maniskill"
    echo "[install_local.sh]   SAPIEN PhysX:    $WORK_DIR/.sapien"
    echo "[install_local.sh]   OpenPI tokenizer: $WORK_DIR/.cache/openpi"
    exit 0
fi

bash "$REPO_DIR/requirements/install.sh" "${INSTALL_ARGS[@]}"

echo "[install_local.sh] Installation finished. Local assets located at:"
echo "[install_local.sh]   Virtual env:     $WORK_DIR/.venv"
echo "[install_local.sh]   uv cache:        $WORK_DIR/.uv_cache"
echo "[install_local.sh]   HF cache:        $WORK_DIR/.hf_home"
echo "[install_local.sh]   LIBERO:          $WORK_DIR/libero"
echo "[install_local.sh]   ManiSkill:       $WORK_DIR/.maniskill"
echo "[install_local.sh]   SAPIEN PhysX:    $WORK_DIR/.sapien"
echo "[install_local.sh]   OpenPI tokenizer: $WORK_DIR/.cache/openpi"
