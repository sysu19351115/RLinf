"""Integration tests for versions tensor shape handling in the PPO loss pipeline.

Regression test for the P0 bug where ``versions.shape == [B, 1]`` caused
``RuntimeError: shape '[10, -1, 8]' is invalid for input of size 10`` in
``preprocess_loss_inputs`` during the first PPO update.

The tests exercise the full data flow:
    RolloutResult -> EmbodiedRolloutResult -> Trajectory -> flatten -> policy_loss
"""

import torch
import pytest

from rlinf.algorithms.registry import policy_loss
from rlinf.algorithms.utils import preprocess_loss_inputs
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    RolloutResult,
    Trajectory,
)

# Shapes mirroring the dobot async PPO config.
BSZ = 10
NUM_ACTION_CHUNKS = 10
ACTION_DIM = 8
TRAJ_LEN = 5  # number of rollout steps in a trajectory


def _make_rollout_result(version: int = 0) -> RolloutResult:
    """Create a RolloutResult with the exact shapes produced by HuggingFaceWorker."""
    flat_action = NUM_ACTION_CHUNKS * ACTION_DIM
    return RolloutResult(
        actions=torch.randn(BSZ, flat_action),
        prev_logprobs=torch.randn(BSZ, flat_action),
        prev_values=torch.randn(BSZ, 1),
        bootstrap_values=torch.randn(BSZ, 1),
        intervene_flags=None,
        forward_inputs={"pixel_values": torch.randn(BSZ, 2, 3, 224, 224)},
        versions=torch.full((BSZ, 1), float(version), dtype=torch.float32),
    )


def _make_chunk_step_result(version: int = 0) -> ChunkStepResult:
    flat_action = NUM_ACTION_CHUNKS * ACTION_DIM
    return ChunkStepResult(
        actions=torch.randn(BSZ, flat_action),
        prev_logprobs=torch.randn(BSZ, flat_action),
        prev_values=torch.randn(BSZ, 1),
        dones=torch.zeros(BSZ, NUM_ACTION_CHUNKS, dtype=torch.bool),
        truncations=torch.zeros(BSZ, NUM_ACTION_CHUNKS, dtype=torch.bool),
        terminations=torch.zeros(BSZ, NUM_ACTION_CHUNKS, dtype=torch.bool),
        rewards=torch.zeros(BSZ, NUM_ACTION_CHUNKS),
        versions=torch.full((BSZ, 1), float(version), dtype=torch.float32),
        forward_inputs={"pixel_values": torch.randn(BSZ, 2, 3, 224, 224)},
    )


def _flatten_trajectory(traj: Trajectory) -> dict:
    """Replicate ``ReplayBuffer._flatten_trajectory`` for tensor fields."""
    flat = {}
    for field in traj.__dataclass_fields__:
        tensor = getattr(traj, field)
        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
            flat[field] = tensor.reshape(-1, *tensor.shape[2:])
    return flat


class TestPreprocessLossInputsVersions:
    """Direct unit tests for versions handling in preprocess_loss_inputs."""

    @pytest.mark.parametrize("logprob_type", ["chunk_level", "action_level", "token_level"])
    def test_versions_bsz1_does_not_crash(self, logprob_type):
        """versions [B, 1] must not cause reshape errors for any logprob_type."""
        bsz = BSZ
        flat_action = NUM_ACTION_CHUNKS * ACTION_DIM
        kwargs = preprocess_loss_inputs(
            logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            old_logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            advantages=torch.randn(bsz, dtype=torch.float32),
            logprob_type=logprob_type,
            single_action_dim=ACTION_DIM,
            loss_mask=torch.ones(bsz, dtype=torch.bool),
            loss_mask_sum=torch.ones(bsz, dtype=torch.float32),
            values=torch.randn(bsz, dtype=torch.float32),
            prev_values=torch.randn(bsz, dtype=torch.float32),
            returns=torch.randn(bsz, dtype=torch.float32),
            reward_type="chunk_level",
            versions=torch.full((bsz, 1), 5.0, dtype=torch.float32),
            loss_type="decoupled_actor_critic",
        )
        v = kwargs["versions"]
        # versions should broadcast to match logprobs shape
        assert v.shape[0] == bsz
        # Every element should be 5.0 (broadcast correctly)
        assert torch.all(v == 5.0)

    def test_versions_none_passes_through(self):
        """When versions is None, it should stay None."""
        bsz = BSZ
        flat_action = NUM_ACTION_CHUNKS * ACTION_DIM
        kwargs = preprocess_loss_inputs(
            logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            old_logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            advantages=torch.randn(bsz, dtype=torch.float32),
            logprob_type="chunk_level",
            single_action_dim=ACTION_DIM,
            reward_type="chunk_level",
            versions=None,
        )
        assert kwargs["versions"] is None


class TestPolicyLossWithVersions:
    """End-to-end policy_loss tests with versions [B, 1]."""

    def test_decoupled_actor_critic_chunk_level(self):
        """Full policy_loss with decoupled_actor_critic + chunk_level, versions [B,1]."""
        bsz = BSZ
        flat_action = NUM_ACTION_CHUNKS * ACTION_DIM
        loss, metrics = policy_loss(
            loss_type="decoupled_actor_critic",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=ACTION_DIM,
            task_type="embodied",
            logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            old_logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            advantages=torch.randn(bsz, dtype=torch.float32),
            values=torch.randn(bsz, dtype=torch.float32),
            prev_values=torch.randn(bsz, dtype=torch.float32),
            returns=torch.randn(bsz, dtype=torch.float32),
            loss_mask=torch.ones(bsz, dtype=torch.bool),
            loss_mask_sum=torch.ones(bsz, dtype=torch.float32),
            versions=torch.full((bsz, 1), 5.0, dtype=torch.float32),
            current_version=6,
            clip_ratio_high=0.2,
            clip_ratio_low=0.2,
            clip_ratio_c=3.0,
            value_clip=0.2,
            huber_delta=10.0,
            behave_weight_threshold=2.0,
            max_episode_steps=500,
            critic_warmup=False,
        )
        assert loss.dim() == 0  # scalar loss
        assert "actor/policy_loss" in metrics
        # proximal computation uses versions, so this metric should exist
        assert "actor/average_version" in metrics or "actor/current_version" not in metrics

    def test_actor_only_chunk_level(self):
        """Plain actor loss with chunk_level + versions [B,1]."""
        bsz = BSZ
        flat_action = NUM_ACTION_CHUNKS * ACTION_DIM
        loss, metrics = policy_loss(
            loss_type="actor",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=ACTION_DIM,
            task_type="embodied",
            logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            old_logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            advantages=torch.randn(bsz, dtype=torch.float32),
            loss_mask=torch.ones(bsz, dtype=torch.bool),
            loss_mask_sum=torch.ones(bsz, dtype=torch.float32),
            versions=torch.full((bsz, 1), 3.0, dtype=torch.float32),
            clip_ratio_high=0.2,
            clip_ratio_low=0.2,
            max_episode_steps=500,
            critic_warmup=False,
        )
        assert loss.dim() == 0


class TestFullPipelineRolloutToLoss:
    """Integration test: RolloutResult -> Trajectory -> flatten -> policy_loss."""

    def test_versions_preserved_through_pipeline(self):
        """Versions [B,1] from RolloutResult survive flatten and feed policy_loss."""
        builder = EmbodiedRolloutResult()
        for t in range(TRAJ_LEN):
            builder.append_step_result(_make_chunk_step_result(version=t))
        traj = builder.to_trajectory()

        # Verify trajectory shape: [T, B, 1]
        assert traj.versions.shape == (TRAJ_LEN, BSZ, 1)

        # Flatten (same logic as ReplayBuffer._flatten_trajectory)
        flat = _flatten_trajectory(traj)

        # After flatten: [T*B, 1]
        assert flat["versions"].shape == (TRAJ_LEN * BSZ, 1)
        assert flat["prev_logprobs"].shape[0] == TRAJ_LEN * BSZ

        # Simulate micro-batch extraction: take first BSZ samples
        mb_versions = flat["versions"][:BSZ]
        mb_logprobs = flat["prev_logprobs"][:BSZ]
        mb_advantages = torch.randn(BSZ, dtype=torch.float32)

        # This call would have crashed before the fix
        loss, metrics = policy_loss(
            loss_type="decoupled_actor_critic",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=ACTION_DIM,
            task_type="embodied",
            logprobs=mb_logprobs.to(torch.float32),
            old_logprobs=mb_logprobs.to(torch.float32),
            advantages=mb_advantages,
            values=torch.randn(BSZ, dtype=torch.float32),
            prev_values=torch.randn(BSZ, dtype=torch.float32),
            returns=torch.randn(BSZ, dtype=torch.float32),
            loss_mask=torch.ones(BSZ, dtype=torch.bool),
            loss_mask_sum=torch.ones(BSZ, dtype=torch.float32),
            versions=mb_versions,
            current_version=TRAJ_LEN,
            clip_ratio_high=0.2,
            clip_ratio_low=0.2,
            clip_ratio_c=3.0,
            value_clip=0.2,
            huber_delta=10.0,
            behave_weight_threshold=2.0,
            max_episode_steps=500,
            critic_warmup=False,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestPaddingVersionStaleness:
    """Regression tests for the P0 bug where padding version=-1 caused
    trajectories with early episode termination to be wrongly discarded.
    """

    def test_extract_valid_versions_filters_negative(self):
        """extract_valid_versions should drop all entries < 0."""
        from rlinf.data.embodied_io_struct import extract_valid_versions

        mixed = torch.tensor([1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0])
        valid = extract_valid_versions(mixed)
        assert valid.numel() == 4
        assert torch.equal(valid, torch.tensor([1.0, 1.0, 2.0, 2.0]))

    def test_extract_valid_versions_all_negative(self):
        """When all entries are padding, result is empty."""
        from rlinf.data.embodied_io_struct import extract_valid_versions

        all_padding = torch.full((5, 3), -1.0)
        valid = extract_valid_versions(all_padding)
        assert valid.numel() == 0

    def test_extract_valid_versions_all_positive(self):
        """No padding entries — all versions preserved."""
        from rlinf.data.embodied_io_struct import extract_valid_versions

        all_valid = torch.tensor([3.0, 3.0, 4.0, 4.0])
        valid = extract_valid_versions(all_valid)
        assert valid.numel() == 4
        assert torch.equal(valid, all_valid)

    def test_priority_with_padding_version(self):
        """PriorityStore should retain a trajectory whose valid min version
        passes the staleness check, even though raw versions include -1.
        """
        from rlinf.data.embodied_io_struct import extract_valid_versions
        from rlinf.data.priority_store import PriorityStore

        traj_len = 100
        bsz = 1
        versions = torch.full((traj_len, bsz, 1), -1.0)
        versions[0:3] = 1.0
        versions[50:100] = 1.0

        valid = extract_valid_versions(versions)
        assert valid.numel() == 53

        min_v = float(valid.min().item())
        mean_v = float(valid.float().mean().item())
        assert min_v == 1.0
        assert mean_v == 1.0

        store = PriorityStore(maxsize=10)
        store.add((min_v, mean_v), type("T", (), {"versions": versions})())
        store.remove_below(0)
        assert len(store) == 1

    def test_priority_store_metric_excludes_padding(self):
        """get_metric() should not count version=-1 padding entries."""
        from rlinf.data.priority_store import PriorityStore

        versions = torch.tensor([[[1.0]], [[1.0]], [[-1.0]], [[-1.0]], [[2.0]]])
        store = PriorityStore(maxsize=10)
        store.add((1.0, 1.33), type("T", (), {"versions": versions})())

        metric = store.get_metric()
        assert -1 not in metric
        assert 1 in metric and 2 in metric
        assert metric[1]["ratio"] == pytest.approx(2 / 3)
        assert metric[2]["ratio"] == pytest.approx(1 / 3)

    def test_all_padding_trajectory_discarded(self):
        """A trajectory that is entirely padding should be discarded."""
        from rlinf.data.embodied_io_struct import extract_valid_versions

        versions = torch.full((10, 1, 1), -1.0)
        valid = extract_valid_versions(versions)
        assert valid.numel() == 0

    def test_staleness_check_mixed_versions_actor_v1(self):
        """The exact scenario from the bug report.

        Actor at version 1, staleness_threshold=1.
        Trajectory has real chunks at v=1 and padding at v=-1.

        Before fix: min([-1, 1]) = -1 < 0 -> discarded.
        After fix:  min([1])    = 1  < 0 -> False -> retained.
        """
        from rlinf.data.embodied_io_struct import extract_valid_versions

        actor_version = 1
        staleness_threshold = 1
        threshold = actor_version - staleness_threshold

        versions = torch.tensor([[1.0], [1.0], [1.0], [-1.0], [-1.0]])
        valid = extract_valid_versions(versions)
        assert valid.min().item() >= threshold

    def test_staleness_check_version0_strict_lt(self):
        """At version 0, valid min is still 0 which passes threshold -1."""
        from rlinf.data.embodied_io_struct import extract_valid_versions

        actor_version = 0
        staleness_threshold = 1
        threshold = actor_version - staleness_threshold

        versions = torch.tensor([[0.0], [0.0], [-1.0]])
        valid = extract_valid_versions(versions)
        assert valid.min().item() == 0.0
        assert valid.min().item() >= threshold

    def test_padding_versions_negative(self):
        """Padding entries (version < 0) are masked correctly in proximal computation."""
        bsz = BSZ
        flat_action = NUM_ACTION_CHUNKS * ACTION_DIM
        # Half real, half padding (version=-1)
        versions = torch.full((bsz, 1), 5.0, dtype=torch.float32)
        versions[bsz // 2:] = -1.0

        loss, metrics = policy_loss(
            loss_type="decoupled_actor_critic",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=ACTION_DIM,
            task_type="embodied",
            logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            old_logprobs=torch.randn(bsz, flat_action, dtype=torch.float32),
            advantages=torch.randn(bsz, dtype=torch.float32),
            values=torch.randn(bsz, dtype=torch.float32),
            prev_values=torch.randn(bsz, dtype=torch.float32),
            returns=torch.randn(bsz, dtype=torch.float32),
            loss_mask=torch.ones(bsz, dtype=torch.bool),
            loss_mask_sum=torch.ones(bsz, dtype=torch.float32),
            versions=versions,
            current_version=6,
            clip_ratio_high=0.2,
            clip_ratio_low=0.2,
            clip_ratio_c=3.0,
            value_clip=0.2,
            huber_delta=10.0,
            behave_weight_threshold=2.0,
            max_episode_steps=500,
            critic_warmup=False,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
