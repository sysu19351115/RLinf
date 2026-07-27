# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contracts for standalone autonomous Dobot HG-DAgger evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from rlinf.envs.realworld.common.wrappers.dobot_keyboard_intervention import (
    DobotKeyboardIntervention,
)
from rlinf.envs.realworld.dobot.dobot_env import DobotEnv, DobotRobotConfig
from rlinf.utils import omega_resolver  # noqa: F401

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


class _Listener:
    def __init__(self):
        self.presses = []
        self.held = set()

    def pop_pressed_keys(self):
        presses, self.presses = self.presses, []
        return presses

    def get_keys(self):
        return frozenset(self.held)

    def get_key(self):
        return next(iter(self.held), None)

    def is_connected(self):
        return True

    def fatal_error(self):
        return None


def _compose_eval(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    monkeypatch.setenv("DOBOT_HG_DAGGER_MODEL_PATH", "/tmp/base_model")
    monkeypatch.setenv("DOBOT_HG_DAGGER_NORM_STATS_PATH", "/tmp/norm.json")
    monkeypatch.setenv("DOBOT_HG_DAGGER_EVAL_CHECKPOINT", "/tmp/checkpoint.pt")
    monkeypatch.setenv("DOBOT_HG_DAGGER_EVAL_CHECKPOINT_ID", "hgdagger-step-0040")
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dobot_hg_dagger_eval")
    OmegaConf.resolve(cfg)
    return cfg


def test_eval_config_is_standalone_autonomous_and_single_robot(monkeypatch):
    cfg = _compose_eval(monkeypatch)

    assert cfg.runner.task_type == "embodied_eval"
    assert cfg.runner.only_eval is True
    assert cfg.runner.val_check_interval == -1
    assert cfg.runner.ckpt_path == "/tmp/checkpoint.pt"
    assert cfg.runner.eval_checkpoint_id == "hgdagger-step-0040"
    assert cfg.env.eval.total_num_envs == 1
    assert cfg.env.eval.use_keyboard_intervention is True
    assert cfg.env.eval.keyboard_intervention.allow_motion_intervention is False
    assert cfg.env.eval.keyboard_intervention.start_in_engage is False
    assert cfg.env.eval.keyboard_intervention.episode_control_mode == "online"
    assert cfg.env.eval.data_collection.enabled is False
    hardware = cfg.cluster.node_groups[0].hardware.configs
    assert len(hardware) == 1


def test_eval_checkpoint_identity_is_required(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    monkeypatch.setenv("DOBOT_HG_DAGGER_MODEL_PATH", "/tmp/base_model")
    monkeypatch.setenv("DOBOT_HG_DAGGER_NORM_STATS_PATH", "/tmp/norm.json")
    monkeypatch.delenv("DOBOT_HG_DAGGER_EVAL_CHECKPOINT", raising=False)
    monkeypatch.delenv("DOBOT_HG_DAGGER_EVAL_CHECKPOINT_ID", raising=False)
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="dobot_hg_dagger_eval")
    with pytest.raises(InterpolationResolutionError):
        OmegaConf.resolve(cfg)


def test_label_only_keyboard_cannot_replace_model_action():
    env = DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
            step_frequency=10_000.0,
        )
    )
    listener = _Listener()
    wrapper = DobotKeyboardIntervention(
        env,
        listener=listener,
        allow_motion_intervention=False,
        episode_control_mode="online",
    )
    wrapper.reset()
    model_action = np.array([0, 0, 0, 1, 0, 0, 0, 0.5], dtype=np.float64)

    listener.presses = ["h"]
    listener.held = {"w", "a"}
    action_out, replaced = wrapper.action(model_action)
    np.testing.assert_array_equal(action_out, model_action)
    assert replaced is False
    assert wrapper._state == "model"

    listener.presses = ["Key.enter"]
    listener.held.clear()
    _, replaced = wrapper.action(model_action)
    assert replaced is False
    assert wrapper._episode_save is True


def test_label_only_mode_rejects_start_in_engage():
    env = DobotEnv(
        DobotRobotConfig(
            is_dummy=True,
            action_mode="cartesian",
            state_mode="pose",
        )
    )
    with pytest.raises(ValueError, match="incompatible"):
        DobotKeyboardIntervention(
            env,
            listener=_Listener(),
            allow_motion_intervention=False,
            start_in_engage=True,
        )
