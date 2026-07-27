# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hardware-free tests for Dobot's host-local ownership lock."""

from __future__ import annotations

import multiprocessing

import pytest

from rlinf.envs.realworld.dobot.dobot_controller import (
    DobotOwnershipError,
    DobotOwnershipLock,
)


def _hold_lock(ip: str, lock_dir: str, ready) -> None:
    lock = DobotOwnershipLock(ip, lock_dir)
    lock.acquire()
    ready.set()
    multiprocessing.Event().wait()


def test_second_owner_fails_and_close_releases(tmp_path):
    first = DobotOwnershipLock("192.168.5.2", tmp_path)
    second = DobotOwnershipLock("192.168.5.2", tmp_path)
    first.acquire()
    try:
        with pytest.raises(DobotOwnershipError, match="already owned"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_lock_is_scoped_by_controller_ip(tmp_path):
    first = DobotOwnershipLock("192.168.5.2", tmp_path)
    other_robot = DobotOwnershipLock("192.168.5.3", tmp_path)
    first.acquire()
    other_robot.acquire()
    other_robot.release()
    first.release()


def test_process_death_does_not_leave_permanent_ownership(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(
        target=_hold_lock,
        args=("192.168.5.2", str(tmp_path), ready),
    )
    process.start()
    assert ready.wait(timeout=10)

    contender = DobotOwnershipLock("192.168.5.2", tmp_path)
    with pytest.raises(DobotOwnershipError):
        contender.acquire()

    process.terminate()
    process.join(timeout=10)
    assert not process.is_alive()

    recovered = DobotOwnershipLock("192.168.5.2", tmp_path)
    recovered.acquire()
    recovered.release()
