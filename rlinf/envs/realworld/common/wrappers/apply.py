# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper-stack builders shared by realworld task factories."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym

from rlinf.envs.realworld.common.wrappers.dobot_keyboard_intervention import (
    DobotKeyboardIntervention,
)
from rlinf.envs.realworld.common.wrappers.dual_gello_joint_intervention import (
    DualGelloJointIntervention,
)
from rlinf.envs.realworld.common.wrappers.euler_obs import Quat2EulerWrapper
from rlinf.envs.realworld.common.wrappers.gello_intervention import (
    GelloIntervention,
)
from rlinf.envs.realworld.common.wrappers.gripper_close import GripperCloseEnv
from rlinf.envs.realworld.common.wrappers.keyboard_eval_control_wrapper import (
    KeyboardEvalControlWrapper,
)
from rlinf.envs.realworld.common.wrappers.keyboard_rlt_policy_switch_wrapper import (
    KeyboardRLTPolicySwitchWrapper,
)
from rlinf.envs.realworld.common.wrappers.keyboard_start_end_wrapper import (
    KeyboardStartEndWrapper,
)
from rlinf.envs.realworld.common.wrappers.pico_intervention import (
    DualFrankaTcpPicoIntervention,
    PicoIntervention,
)
from rlinf.envs.realworld.common.wrappers.relative_frame import RelativeFrame
from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
)
from rlinf.envs.realworld.common.wrappers.so101_keyboard_intervention import (
    SO101KeyboardIntervention,
)
from rlinf.envs.realworld.common.wrappers.spacemouse_intervention import (
    SpacemouseIntervention,
)


def _load_dexhand_intervention():
    """Import DexHandIntervention only when dex-hand teleop is requested."""
    try:
        from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
            DexHandIntervention,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "rlinf_dexhand":
            raise ModuleNotFoundError(
                "DexHandIntervention requires optional dependency "
                "'rlinf_dexhand'. Install it before enabling "
                "dexterous-hand teleoperation."
            ) from exc
        raise
    return DexHandIntervention


def _validate_teleop_mode(**modes: bool) -> None:
    active_modes = [name for name, enabled in modes.items() if bool(enabled)]
    if len(active_modes) > 1:
        raise ValueError(
            "Only one teleop mode can be active at a time. "
            f"Active modes: {', '.join(active_modes)}."
        )


def _apply_keyboard_wrapper(env: gym.Env, mode: Optional[str]) -> gym.Env:
    config = env.get_wrapper_attr("config")
    if config.is_dummy or not mode:
        return env
    if mode == "multi_stage":
        return KeyboardRewardDoneMultiStageWrapper(env)
    if mode == "single_stage":
        return KeyboardRewardDoneWrapper(env)
    if mode == "start_end":
        return KeyboardStartEndWrapper(env)
    if mode == "eval_control":
        return KeyboardEvalControlWrapper(env)
    if mode == "rlt_policy_switch":
        return KeyboardRLTPolicySwitchWrapper(env)
    return env


def apply_single_arm_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for single-arm realworld envs (franka single, xsquare)."""
    end_effector_type = str(
        getattr(getattr(env, "config", None), "end_effector_type", "franka_gripper")
    )
    is_dex_hand = end_effector_type.endswith("hand")

    no_gripper = cfg.get("no_gripper", True)
    if no_gripper and not is_dex_hand:
        env = GripperCloseEnv(env)

    use_spacemouse = cfg.get("use_spacemouse", True)
    use_gello = cfg.get("use_gello", False)
    use_pico = cfg.get("use_pico", False)
    _validate_teleop_mode(
        use_spacemouse=use_spacemouse,
        use_gello=use_gello,
        use_pico=use_pico,
    )

    gripper_enabled = not no_gripper

    if not env.config.is_dummy and use_spacemouse:
        if is_dex_hand:
            glove_cfg = cfg.get("glove_config", {})
            DexHandIntervention = _load_dexhand_intervention()
            env = DexHandIntervention(
                env,
                left_port=glove_cfg.get("left_port", "/dev/ttyACM0"),
                right_port=glove_cfg.get("right_port", None),
                glove_frequency=glove_cfg.get("frequency", 60),
                glove_config_file=glove_cfg.get("config_file", None),
            )
        else:
            env = SpacemouseIntervention(env, gripper_enabled=gripper_enabled)

    if not env.config.is_dummy and use_gello:
        if is_dex_hand:
            raise ValueError("use_gello=True is not supported for ruiyan_hand.")
        gello_port = cfg.get("gello_port", None)
        if gello_port is None:
            raise ValueError(
                "use_gello=True requires 'gello_port' in the env config "
                "(e.g. env.eval.gello_port)."
            )
        env = GelloIntervention(env, port=gello_port, gripper_enabled=gripper_enabled)

    if not env.config.is_dummy and use_pico:
        if is_dex_hand:
            raise ValueError("use_pico=True is not supported for dexterous hands.")
        pico_cfg = dict(cfg.get("pico", {}))
        env = PicoIntervention(env, gripper_enabled=gripper_enabled, **pico_cfg)

    env = _apply_keyboard_wrapper(env, cfg.get("keyboard_reward_wrapper", None))

    if cfg.get("use_relative_frame", True):
        env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    return env


def apply_so101_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for SO101 bimanual real-world envs.

    Adds optional keyboard-based 6D end-effector intervention. The underlying
    :class:`SO101Env` must expose ``get_joint_positions()``.
    """
    config = env.get_wrapper_attr("config")
    use_keyboard = cfg.get("use_keyboard_intervention", False)
    active_in_dummy = not config.is_dummy or cfg.get("use_intervention_in_dummy", False)

    if use_keyboard and active_in_dummy:
        kcfg = cfg.get("keyboard_intervention", {})
        env = SO101KeyboardIntervention(
            env,
            active_arm=kcfg.get("active_arm", "left"),
            position_delta=float(kcfg.get("position_delta", 0.005)),
            rotation_delta=float(kcfg.get("rotation_delta", 0.05)),
            gripper_delta=float(kcfg.get("gripper_delta", 5.0)),
            urdf_path=kcfg.get("urdf_path", None),
            end_effector_link=kcfg.get("end_effector_link", "gripper_frame_link"),
            toggle_key=kcfg.get("toggle_key", "h"),
            model_key=kcfg.get("model_key", "m"),
            quit_keys=tuple(kcfg.get("quit_keys", ("Key.esc",))),
            switch_arm_key=kcfg.get("switch_arm_key", "Tab"),
        )

    env = _apply_keyboard_wrapper(env, cfg.get("keyboard_reward_wrapper", None))
    return env


def apply_dobot_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for Dobot CR5AF real-world envs.

    Adds optional keyboard-based 6D Cartesian end-effector intervention.
    The underlying :class:`DobotEnv` must be in ``state_mode='pose'`` /
    ``action_mode='cartesian'`` and expose ``get_pose_state()`` /
    ``reset_servo_smoothing()``.
    """
    config = env.get_wrapper_attr("config")
    use_keyboard = cfg.get("use_keyboard_intervention", False)
    active_in_dummy = not config.is_dummy or cfg.get("use_intervention_in_dummy", False)

    if use_keyboard and active_in_dummy:
        if config.action_mode != "cartesian" or config.state_mode != "pose":
            raise ValueError(
                "DobotKeyboardIntervention requires action_mode='cartesian' and "
                f"state_mode='pose', got action_mode={config.action_mode!r}, "
                f"state_mode={config.state_mode!r}."
            )
        kcfg = cfg.get("keyboard_intervention", {})
        env = DobotKeyboardIntervention(
            env,
            position_delta=float(kcfg.get("position_delta", 0.002)),
            rotation_delta=float(kcfg.get("rotation_delta", 0.02)),
            gripper_delta=float(kcfg.get("gripper_delta", 0.05)),
            workspace_low=kcfg.get("workspace_low"),
            workspace_high=kcfg.get("workspace_high"),
            base_frame_euler_deg=kcfg.get("base_frame_euler_deg", [0.0, 0.0, 0.0]),
            toggle_key=kcfg.get("toggle_key", "h"),
            model_key=kcfg.get("model_key", "m"),
            done_key=kcfg.get("done_key", "Key.enter"),
            abort_key=kcfg.get("abort_key", "Key.backspace"),
            quit_keys=tuple(kcfg.get("quit_keys", ("Key.esc",))),
            start_in_engage=bool(kcfg.get("start_in_engage", False)),
            episode_control_mode=kcfg.get("episode_control_mode", "collector"),
            wait_for_start_on_reset=bool(kcfg.get("wait_for_start_on_reset", False)),
            start_key=kcfg.get("start_key", "y"),
            start_gate_timeout_s=kcfg.get("start_gate_timeout_s", None),
        )

    env = _apply_keyboard_wrapper(env, cfg.get("keyboard_reward_wrapper", None))
    return env


def apply_dual_franka_joint_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    config = env.get_wrapper_attr("config")
    if cfg.get("no_gripper", True):
        # No DualGripperCloseEnv yet, so a 12D action would blow up as reshape(2,7).
        raise NotImplementedError(
            "no_gripper=True not supported for dual-arm envs (no DualGripperCloseEnv)."
        )

    use_pico = cfg.get("use_pico", False)
    use_gello_joint = cfg.get("use_gello_joint", False)
    if cfg.get("use_spacemouse", False) or cfg.get("use_gello", False):
        raise ValueError(
            "Dual-arm Franka envs do not support use_spacemouse=True or "
            "use_gello=True. Use use_gello_joint=True for GELLO-joint teleop "
            "or use_pico=True for dual-arm PICO teleop."
        )
    _validate_teleop_mode(use_gello_joint=use_gello_joint, use_pico=use_pico)

    if not config.is_dummy and use_gello_joint:
        left_port = cfg.get("left_gello_port", None)
        right_port = cfg.get("right_gello_port", None)
        if left_port is None or right_port is None:
            raise ValueError(
                "use_gello_joint=True requires both "
                "'left_gello_port' and 'right_gello_port' in the env config."
            )
        env = DualGelloJointIntervention(
            env,
            left_port=left_port,
            right_port=right_port,
            gripper_enabled=True,
            use_delta=getattr(config, "joint_action_mode", None) == "delta",
            action_scale=getattr(config, "joint_action_scale", 0.1),
            direct_stream=getattr(config, "teleop_direct_stream", False),
            stream_period=cfg.get("gello_joint_stream_period", 0.001),
        )

    if not config.is_dummy and use_pico:
        if getattr(env.unwrapped, "PER_ARM_ACTION_DIM", None) != 10:
            raise ValueError(
                "use_pico=True for dual-arm Franka is implemented for "
                "DualFrankaTcpEnv-v1 only. Use env/realworld_dual_franka_tcp_rot6d."
            )
        pico_cfg = dict(cfg.get("pico", {}))
        env = DualFrankaTcpPicoIntervention(
            env,
            gripper_enabled=True,
            **pico_cfg,
        )

    env = _apply_keyboard_wrapper(env, cfg.get("keyboard_reward_wrapper", None))
    return env
