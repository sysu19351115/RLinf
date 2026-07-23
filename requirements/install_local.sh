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
CPU_ONLY=0
INSTALL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        --cpu-only)
            CPU_ONLY=1
            shift
            ;;
        *)
            INSTALL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Detect NVIDIA Blackwell (RTX 50 series / sm_120) GPUs and automatically upgrade
# PyTorch to a compatible version. This only applies when the user has not
# explicitly requested a torch version or CUDA backend.
# Skip this logic in --cpu-only mode since we want CPU torch, not CUDA torch.
_detect_blackwell_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        case "$gpu_name" in
            *"RTX 50"*|*"RTX 5090"*|*"RTX 5080"*|*"RTX 5070"*|*"RTX 5060"*)
                echo "1"
                return
                ;;
        esac
    fi
    echo "0"
}

if [[ "$CPU_ONLY" -ne 1 ]]; then

_TORCH_VERSION_EXPLICIT=0
_TORCH_BACKEND_EXPLICIT=0
for arg in "${INSTALL_ARGS[@]}"; do
    [[ "$arg" == "--torch" ]] && _TORCH_VERSION_EXPLICIT=1
    [[ "$arg" == "--platform" ]] && _TORCH_BACKEND_EXPLICIT=1
done

if [[ "$_TORCH_VERSION_EXPLICIT" -eq 0 && "$_TORCH_BACKEND_EXPLICIT" -eq 0 && -z "${UV_TORCH_BACKEND:-}" ]]; then
    if [[ "$(_detect_blackwell_gpu)" == "1" ]]; then
        echo "[install_local.sh] Detected NVIDIA Blackwell GPU (RTX 50 series). Auto-selecting torch 2.7.0 + cu128."
        INSTALL_ARGS+=("--torch" "2.7.0")
        export UV_TORCH_BACKEND="cu128"
    fi
fi

fi  # end CPU_ONLY guard

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
VENV_DIR="${VENV_DIR:-$WORK_DIR/.venv}"

# Extract --env from the collected args so we know which asset checks apply.
_ENV_NAME=""
for ((_i=0; _i<${#INSTALL_ARGS[@]}; _i++)); do
    if [[ "${INSTALL_ARGS[$_i]}" == "--env" ]] && (( _i+1 < ${#INSTALL_ARGS[@]} )); then
        _ENV_NAME="${INSTALL_ARGS[$_i+1]}"
        break
    fi
done

# Environments that pull large asset directories.  When _ENV_NAME is empty (not
# specified) we fall back to checking everything — the original behaviour.
_NEED_LIBERO=("libero" "maniskill_libero" "liberopro" "liberoplus")
_NEED_MANISKILL=("maniskill_libero")
_NEED_SAPIEN=("maniskill_libero")

_array_contains() {
    local needle="$1"; shift
    local item
    for item in "$@"; do [[ "$item" == "$needle" ]] && return 0; done
    return 1
}

# Verify that the env-specific Python package is actually importable inside the
# venv.  Lightweight envs that don't pull large asset directories rely on this
# check instead of the asset checks above.
_env_package_installed() {
    case "$_ENV_NAME" in
        gym_aloha)
            "$WORK_DIR/.venv/bin/python" -c "import gym_aloha" 2>/dev/null || return 1
            ;;
        rebot)
            "$WORK_DIR/.venv/bin/python" -c \
                "import motorbridge, pinocchio" 2>/dev/null || return 1
            ;;
        so101)
            "$WORK_DIR/.venv/bin/python" -c \
                "import lerobot, pinocchio" 2>/dev/null || return 1
            ;;
        dobot)
            "$WORK_DIR/.venv/bin/python" -c \
                "import cv2, motorbridge, scipy" 2>/dev/null || return 1
            ;;
    esac
    return 0
}

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
    if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_LIBERO[@]}"; then
        [[ -f "$WORK_DIR/libero/setup.py" || -f "$WORK_DIR/libero/pyproject.toml" ]] || return 1
    fi
    # ManiSkill assets must be non-empty.
    if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_MANISKILL[@]}"; then
        dir_has_content "$WORK_DIR/.maniskill" || return 1
    fi
    # SAPIEN PhysX libraries must be non-empty.
    if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_SAPIEN[@]}"; then
        dir_has_content "$WORK_DIR/.sapien/physx/105.1-physx-5.3.1.patch0" || return 1
    fi
    # OpenPI tokenizer marker. Note: hf download puts it under big_vision/.
    [[ -f "$WORK_DIR/.cache/openpi/big_vision/paligemma_tokenizer.model" ]] || return 1
    # Lightweight envs: verify the env-specific package is importable.
    _env_package_installed || return 1
    return 0
}

# ======================= CPU-ONLY MODE =======================
# In --cpu-only mode we perform a minimal installation suitable for a
# real-world robot control node (env worker only).  This bypasses
# install.sh entirely: no training deps, no model packages, no GPU
# libraries, and torch is resolved as a CPU build.
if [[ "$CPU_ONLY" -eq 1 ]]; then
    echo "[install_local.sh] ============================================================"
    echo "[install_local.sh] CPU-only mode: installing minimal env-worker dependencies."
    echo "[install_local.sh] ============================================================"

    PYTHON_VERSION="${PYTHON_VERSION:-3.11.14}"

    if [[ -z "$_ENV_NAME" ]]; then
        echo "[install_local.sh] WARNING: --env not specified. Only core + realworld-env"
        echo "[install_local.sh]          deps will be installed. If your robot requires"
        echo "[install_local.sh]          additional packages (e.g. --env franka), pass it."
    fi

    export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"

    if ! command -v uv &>/dev/null; then
        if command -v pip &>/dev/null && pip install uv 2>/dev/null; then
            :
        else
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi

    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"

    _EXTRA_ARGS=("--extra" "realworld-env")
    if [[ -n "$_ENV_NAME" ]]; then
        case "$_ENV_NAME" in
            franka)           _EXTRA_ARGS+=("--extra" "franka") ;;
            xsquare_turtle2)  _EXTRA_ARGS+=("--extra" "xsquare_turtle2") ;;
            gim_arm)          _EXTRA_ARGS+=("--extra" "gim_arm") ;;
            rebot)            _EXTRA_ARGS+=("--extra" "rebot") ;;
            so101)            _EXTRA_ARGS+=("--extra" "so101") ;;
            dobot)            _EXTRA_ARGS+=("--extra" "dobot") ;;
            frankasim)        _EXTRA_ARGS+=("--extra" "franka") ;;
            *)
                echo "[install_local.sh] WARNING: --env '$_ENV_NAME' has no matching"
                echo "[install_local.sh]          pyproject.toml extra; skipping." >&2
                ;;
        esac
    fi

    echo "[install_local.sh] Running: uv sync ${_EXTRA_ARGS[*]} --no-install-project"
    uv sync "${_EXTRA_ARGS[@]}" --no-install-project

    echo "[install_local.sh] Installing RLinf (editable) into the venv..."
    pip install -e .

    echo "[install_local.sh] ============================================================"
    echo "[install_local.sh] CPU-only installation complete."
    echo "[install_local.sh]   Venv:   source $VENV_DIR/bin/activate"
    echo "[install_local.sh] ============================================================"
    exit 0
fi

if looks_complete; then
    echo "[install_local.sh] Existing installation looks complete. Skipping full install.sh to avoid"
    echo "[install_local.sh] repeated dependency reconciliation. Use --force to run the full install anyway."
    echo "[install_local.sh] Installation finished. Local assets located at:"
    echo "[install_local.sh]   Virtual env:     $WORK_DIR/.venv"
    echo "[install_local.sh]   uv cache:        $WORK_DIR/.uv_cache"
    echo "[install_local.sh]   HF cache:        $WORK_DIR/.hf_home"
    if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_LIBERO[@]}"; then
        echo "[install_local.sh]   LIBERO:          $WORK_DIR/libero"
    fi
    if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_MANISKILL[@]}"; then
        echo "[install_local.sh]   ManiSkill:       $WORK_DIR/.maniskill"
    fi
    if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_SAPIEN[@]}"; then
        echo "[install_local.sh]   SAPIEN PhysX:    $WORK_DIR/.sapien"
    fi
    echo "[install_local.sh]   OpenPI tokenizer: $WORK_DIR/.cache/openpi"
    exit 0
fi

bash "$REPO_DIR/requirements/install.sh" "${INSTALL_ARGS[@]}"

echo "[install_local.sh] Installation finished. Local assets located at:"
echo "[install_local.sh]   Virtual env:     $WORK_DIR/.venv"
echo "[install_local.sh]   uv cache:        $WORK_DIR/.uv_cache"
echo "[install_local.sh]   HF cache:        $WORK_DIR/.hf_home"
if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_LIBERO[@]}"; then
    echo "[install_local.sh]   LIBERO:          $WORK_DIR/libero"
fi
if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_MANISKILL[@]}"; then
    echo "[install_local.sh]   ManiSkill:       $WORK_DIR/.maniskill"
fi
if [[ -z "$_ENV_NAME" ]] || _array_contains "$_ENV_NAME" "${_NEED_SAPIEN[@]}"; then
    echo "[install_local.sh]   SAPIEN PhysX:    $WORK_DIR/.sapien"
fi
echo "[install_local.sh]   OpenPI tokenizer: $WORK_DIR/.cache/openpi"
