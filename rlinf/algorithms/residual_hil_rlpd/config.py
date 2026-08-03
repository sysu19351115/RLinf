# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration contract for the residual HIL-RLPD algorithm.

Every dangerous or task-specific setting must live in the YAML config and be
validated here; the implementation must not silently fill in safety defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_REQUIRED_RLPD_KEYS = (
    "translation_data_limit_m",
    "rotation_data_limit_deg",
    "translation_policy_limit_m",
    "rotation_policy_limit_deg",
    "demo_ratio",
    "min_demo_size",
    "utd_ratio",
    "max_update_backlog",
    "critic_only_updates",
    "residual_scale_ramp_updates",
    "base_only_collect_steps",
    "num_q_heads",
    "num_q_sample",
    "backup_entropy",
    "alpha_arm_init",
    "alpha_gripper_init",
    "max_online_transitions",
    "max_demo_transitions",
    "max_online_bytes",
    "max_demo_bytes",
    "memory_high_watermark",
    "policy_lag_warn_threshold",
    "policy_lag_reject_threshold",
    "gripper_enable_after_updates",
    "gripper_max_switches_per_chunk",
    "gripper_min_hold_steps",
    "gripper_debounce_chunks",
    "safety_workspace_min_m",
    "safety_workspace_max_m",
    "safety_max_translation_delta_m",
    "safety_max_rotation_delta_deg",
    "safety_hold_chunks",
)


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    """OmegaConf/DictConfig-agnostic dotted getter."""
    node: Any = cfg
    for part in path.split("."):
        if node is None:
            return default
        if hasattr(node, "get"):
            node = node.get(part, default)
        elif isinstance(node, dict):
            node = node.get(part, default)
        else:
            return default
    return node


def _require(cfg: Any, path: str, description: str) -> Any:
    value = _get(cfg, path, None)
    if value is None:
        raise ValueError(
            f"residual_hil_rlpd requires {path} ({description}); "
            "it must be set explicitly in the YAML config."
        )
    return value


def _require_vector(cfg: Any, path: str, size: int) -> list[float]:
    value = _require(cfg, path, f"expected a {size}-element vector")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a list, got {type(value).__name__}")
    if len(value) != size:
        raise ValueError(f"{path} must have {size} elements, got {len(value)}")
    return [float(v) for v in value]


def validate_residual_hil_rlpd_config(cfg: Any) -> None:
    """Validate the full residual HIL-RLPD contract.

    Args:
        cfg: The fully composed Hydra config.

    Raises:
        ValueError: On any contract violation, with an actionable message.
    """
    # ── Scope: Dobot cartesian/pose single real environment only ──
    run_id = str(_get(cfg, "run_id", ""))
    if not run_id:
        raise ValueError(
            "residual_hil_rlpd requires a non-empty top-level run_id; every "
            "worker validates message envelopes against it (P2-6)."
        )
    if _get(cfg, "dobot", None) is None:
        raise ValueError(
            "residual_hil_rlpd only supports the Dobot real-world setup; "
            "top-level 'dobot' config block is missing."
        )
    for mode in ("train", "eval"):
        env_cfg = _get(cfg, f"env.{mode}", None)
        if env_cfg is None:
            raise ValueError(f"env.{mode} is missing")
        action_mode = _get(env_cfg, "override_cfg.action_mode", None)
        state_mode = _get(env_cfg, "override_cfg.state_mode", None)
        if action_mode != "cartesian":
            raise ValueError(
                "residual_hil_rlpd requires action_mode=cartesian "
                f"(env.{mode}), got {action_mode!r}"
            )
        if state_mode != "pose":
            raise ValueError(
                "residual_hil_rlpd requires state_mode=pose "
                f"(env.{mode}), got {state_mode!r}"
            )
        if int(_get(env_cfg, "total_num_envs", 0)) != 1:
            raise ValueError(
                "residual_hil_rlpd supports a single real environment "
                f"(env.{mode}.total_num_envs == 1)"
            )

    # ── Algorithm / rollout contract ──
    adv_type = _get(cfg, "algorithm.adv_type", None)
    if adv_type != "embodied_sac":
        raise ValueError(
            "residual_hil_rlpd requires algorithm.adv_type=embodied_sac, "
            f"got {adv_type!r}"
        )

    actor_chunks = _get(cfg, "actor.model.num_action_chunks", None)
    rollout_chunks = _get(cfg, "rollout.model.num_action_chunks", None)
    if actor_chunks != 10 or rollout_chunks != 10:
        raise ValueError(
            "residual_hil_rlpd V1 requires num_action_chunks == 10 for both "
            f"actor and rollout; got actor={actor_chunks}, rollout={rollout_chunks}"
        )

    if not bool(_get(cfg, "rollout.collect_transitions", False)):
        raise ValueError("residual_hil_rlpd requires rollout.collect_transitions=True")

    global_batch = int(_get(cfg, "actor.global_batch_size", 0))
    if global_batch <= 0 or global_batch % 2 != 0:
        raise ValueError(
            "residual_hil_rlpd requires an even actor.global_batch_size for "
            f"strict 50/50 sampling; got {global_batch}"
        )

    # ── Base policy must stay frozen ──
    base_trainable = _get(cfg, "actor.model.base_policy.trainable", False)
    if base_trainable:
        raise ValueError(
            "residual_hil_rlpd requires the Pi0.5 base policy to stay frozen "
            "(actor.model.base_policy.trainable=False); direct VLA fine-tuning "
            "is explicitly out of scope."
        )

    # ── Keyboard intervention safety ──
    if not bool(_get(cfg, "env.train.keyboard_intervention.safe_model_handoff", True)):
        raise ValueError(
            "residual_hil_rlpd requires safe_model_handoff=True for training"
        )
    if bool(
        _get(cfg, "env.eval.keyboard_intervention.allow_motion_intervention", False)
    ):
        raise ValueError(
            "autonomous evaluation must set "
            "env.eval.keyboard_intervention.allow_motion_intervention=False"
        )

    # ── Algorithm parameters must be explicit ──
    rlpd_cfg = _get(cfg, "algorithm.residual_hil_rlpd", None)
    if rlpd_cfg is None:
        raise ValueError(
            "residual_hil_rlpd requires an algorithm.residual_hil_rlpd block"
        )
    # The workspace box is robot/calibration-specific; it is required whenever
    # the check is active.  Operators may consciously disable just this check
    # (the SDK-side per-step jump guard and policy limits remain active) with
    # ``safety_workspace_check_enabled: False``.
    workspace_check_enabled = bool(
        _get(
            cfg,
            "algorithm.residual_hil_rlpd.safety_workspace_check_enabled",
            True,
        )
    )
    for key in _REQUIRED_RLPD_KEYS:
        if (
            not workspace_check_enabled
            and key in {"safety_workspace_min_m", "safety_workspace_max_m"}
        ):
            continue
        _require(cfg, f"algorithm.residual_hil_rlpd.{key}", "explicit YAML value")

    trans_data = _require_vector(
        cfg, "algorithm.residual_hil_rlpd.translation_data_limit_m", 3
    )
    rot_data = _require_vector(
        cfg, "algorithm.residual_hil_rlpd.rotation_data_limit_deg", 3
    )
    trans_policy = _require_vector(
        cfg, "algorithm.residual_hil_rlpd.translation_policy_limit_m", 3
    )
    rot_policy = _require_vector(
        cfg, "algorithm.residual_hil_rlpd.rotation_policy_limit_deg", 3
    )
    if any(p > d for p, d in zip(trans_policy, trans_data)):
        raise ValueError(
            "translation_policy_limit_m must be <= translation_data_limit_m "
            "on every axis"
        )
    if any(p > d for p, d in zip(rot_policy, rot_data)):
        raise ValueError(
            "rotation_policy_limit_deg must be <= rotation_data_limit_deg on every axis"
        )

    demo_ratio = float(_get(cfg, "algorithm.residual_hil_rlpd.demo_ratio", 0.5))
    if not 0.0 < demo_ratio < 1.0:
        raise ValueError(f"demo_ratio must be in (0, 1), got {demo_ratio}")
    if bool(_get(cfg, "algorithm.residual_hil_rlpd.backup_entropy", False)):
        raise ValueError(
            "backup_entropy=True is not yet supported; it requires its own "
            "validated target formula before it can be enabled"
        )

    # ── Replay capacity / memory budget (P2-1) ──
    min_demo_size = int(_get(cfg, "algorithm.residual_hil_rlpd.min_demo_size", 1))
    if min_demo_size < 1:
        raise ValueError(f"min_demo_size must be >= 1, got {min_demo_size}")
    max_online_transitions = int(
        _get(cfg, "algorithm.residual_hil_rlpd.max_online_transitions", 20_000)
    )
    max_demo_transitions = int(
        _get(cfg, "algorithm.residual_hil_rlpd.max_demo_transitions", 5_000)
    )
    max_online_bytes = float(
        _get(cfg, "algorithm.residual_hil_rlpd.max_online_bytes", 40e9)
    )
    max_demo_bytes = float(
        _get(cfg, "algorithm.residual_hil_rlpd.max_demo_bytes", 10e9)
    )
    if max_online_transitions < 1 or max_demo_transitions < 1:
        raise ValueError("max_online/demo_transitions must be >= 1")
    if max_online_bytes <= 0 or max_demo_bytes <= 0:
        raise ValueError("max_online/demo_bytes must be positive")
    watermark = float(
        _get(cfg, "algorithm.residual_hil_rlpd.memory_high_watermark", 0.9)
    )
    if not 0.0 < watermark <= 1.0:
        raise ValueError(f"memory_high_watermark must be in (0, 1], got {watermark}")

    # ── Policy staleness bounds (P2-3) ──
    lag_warn = int(
        _get(cfg, "algorithm.residual_hil_rlpd.policy_lag_warn_threshold", 50)
    )
    lag_reject = int(
        _get(cfg, "algorithm.residual_hil_rlpd.policy_lag_reject_threshold", 500)
    )
    if lag_warn < 1 or lag_reject < lag_warn:
        raise ValueError(
            "policy_lag thresholds must satisfy 1 <= warn <= reject, "
            f"got warn={lag_warn}, reject={lag_reject}"
        )

    # ── Gripper enable / debounce (P2-4) ──
    gripper_enable_after = int(
        _get(cfg, "algorithm.residual_hil_rlpd.gripper_enable_after_updates", 1500)
    )
    max_switches = int(
        _get(cfg, "algorithm.residual_hil_rlpd.gripper_max_switches_per_chunk", 2)
    )
    min_hold = int(
        _get(cfg, "algorithm.residual_hil_rlpd.gripper_min_hold_steps", 5)
    )
    debounce = int(
        _get(cfg, "algorithm.residual_hil_rlpd.gripper_debounce_chunks", 2)
    )
    allow_force_open = _get(
        cfg, "algorithm.residual_hil_rlpd.gripper_allow_force_open", True
    )
    if not isinstance(allow_force_open, bool):
        raise ValueError("gripper_allow_force_open must be a bool")
    if (
        gripper_enable_after < 0
        or max_switches < 1
        or min_hold < 0
        or debounce < 0
    ):
        raise ValueError(
            "gripper enable/debounce params must be non-negative "
            "(max_switches_per_chunk >= 1)"
        )

    # ── Safety barrier limits (P2-5) ──
    if workspace_check_enabled:
        ws_min = _require_vector(
            cfg, "algorithm.residual_hil_rlpd.safety_workspace_min_m", 3
        )
        ws_max = _require_vector(
            cfg, "algorithm.residual_hil_rlpd.safety_workspace_max_m", 3
        )
        if any(a >= b for a, b in zip(ws_min, ws_max)):
            raise ValueError(
                "safety_workspace_min_m must be strictly below "
                "safety_workspace_max_m on every axis"
            )
    if float(
        _get(cfg, "algorithm.residual_hil_rlpd.safety_max_translation_delta_m", 0.02)
    ) <= 0.0 or float(
        _get(cfg, "algorithm.residual_hil_rlpd.safety_max_rotation_delta_deg", 10.0)
    ) <= 0.0:
        raise ValueError("safety_max_translation/rotation delta must be positive")
    if int(_get(cfg, "algorithm.residual_hil_rlpd.safety_hold_chunks", 3)) < 0:
        raise ValueError("safety_hold_chunks must be >= 0")


def get_residual_rlpd_cfg(cfg: Any) -> Any:
    """Return the validated ``algorithm.residual_hil_rlpd`` config block."""
    block = _get(cfg, "algorithm.residual_hil_rlpd", None)
    if block is None:
        raise ValueError("algorithm.residual_hil_rlpd config block is missing")
    return block


def _block_vector(block: Any, key: str, size: int) -> list[float]:
    value = _get(block, key, None)
    if value is None or not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a {size}-element list")
    if len(value) != size:
        raise ValueError(f"{key} must have {size} elements, got {len(value)}")
    return [float(v) for v in value]


def build_residual_codec(cfg: Any):
    """Single source of truth for the residual codec (P2-8): data limits come
    only from the validated YAML block; nothing is hardcoded at call sites."""
    from rlinf.algorithms.residual_hil_rlpd.action_codec import ResidualCodec

    block = get_residual_rlpd_cfg(cfg)
    return ResidualCodec(
        translation_scale_m=tuple(
            _block_vector(block, "translation_data_limit_m", 3)
        ),
        rotation_scale_deg=tuple(_block_vector(block, "rotation_data_limit_deg", 3)),
    )
