from unittest.mock import Mock

import pytest
import torch
import torch.distributed.distributed_c10d as distributed_c10d

from rlinf.scheduler.collective.async_work import AsyncFuncWork
from rlinf.scheduler.collective.collective_group import CollectiveGroup
from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup


def test_broadcast_reraises_process_group_failure(monkeypatch):
    process_group = Mock()
    process_group.broadcast.side_effect = RuntimeError("connection closed by peer")
    monkeypatch.setattr(distributed_c10d, "_rank_not_in_group", lambda group: False)
    monkeypatch.setattr(distributed_c10d, "get_group_rank", lambda group, rank: rank)
    monkeypatch.setattr(
        torch.distributed,
        "_get_process_group_name",
        lambda group: "test-group",
    )

    multi_channel_group = object.__new__(MultiChannelProcessGroup)
    multi_channel_group._cur_rank = 1
    multi_channel_group._logger = Mock()

    with pytest.raises(RuntimeError, match="connection closed by peer"):
        multi_channel_group._broadcast(
            torch.zeros(1, dtype=torch.long),
            src=0,
            group=process_group,
        )


def test_async_work_propagates_failure_and_skips_callbacks():
    def fail():
        raise RuntimeError("collective failed")

    callback = Mock()
    work = AsyncFuncWork(fail)
    chained_work = work.then(callback)

    work(None)

    with pytest.raises(RuntimeError, match="collective failed"):
        work.wait()
    with pytest.raises(RuntimeError, match="collective failed"):
        chained_work.wait()
    callback.assert_not_called()


@pytest.mark.parametrize("metadata_size", [0, -1, 64 * 1024 * 1024 + 1])
def test_tensor_list_broadcast_rejects_invalid_metadata_size(metadata_size):
    collective_group = object.__new__(CollectiveGroup)
    collective_group._rank = 1
    collective_group._broadcast = lambda tensor, **kwargs: tensor.fill_(metadata_size)

    with pytest.raises(RuntimeError, match="Invalid tensor-list metadata size"):
        collective_group._broadcast_tensor_list(
            tensors=None,
            comm_id=0,
            src_rank=0,
        )
