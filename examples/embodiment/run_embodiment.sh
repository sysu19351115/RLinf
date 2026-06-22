#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_PATH}/.cache/openpi}"
export HF_HOME="${HF_HOME:-${REPO_PATH}/.hf_home}"

export MUJOCO_GL=${MUJOCO_GL:-"egl"} # osmesa cpu render
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
# Put the LIBERO package directory first so the editable install is not shadowed
# by the namespace package at ${REPO_PATH}/libero.
export PYTHONPATH="${REPO_PATH}/libero/libero:${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH"

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

# POLARIS dataset
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"maniskill_ppo_openvlaoft"}
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

case "$ROBOT_PLATFORM" in
    LIBERO)
        # Libero variant: standard, pro, plus
        export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
        if [ "$LIBERO_TYPE" == "pro" ]; then
            export LIBERO_PERTURBATION="all"  # all,swap,object,lan
            echo "Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
        elif [ "$LIBERO_TYPE" == "plus" ]; then
            export LIBERO_SUFFIX="all"
            echo "Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
        else
            echo "Mode: Standard LIBERO"
        fi
        ;;
    ALOHA)
        # gym_aloha MuJoCo simulation — same render backend as LIBERO
        echo "Mode: ALOHA (gym_aloha simulation)"
        ;;
    *)
        ;;
esac

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

# Prefer the repository's local venv if it exists.
if [ -x "${REPO_PATH}/.venv/bin/python" ]; then
    PYTHON_CMD="${REPO_PATH}/.venv/bin/python"
else
    PYTHON_CMD="$(which python)"
fi
echo "Using Python at ${PYTHON_CMD}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="${PYTHON_CMD} ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
