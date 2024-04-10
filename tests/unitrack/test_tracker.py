r"""
Tests for ``unitrack.tracker``.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from unitrack import (
    MultiStageTracker,
    TrackletMemory,
    assignment,
    costs,
    stages,
    states,
)


def test_tracker():
    dtype = torch.float32
    frames = [
        {
            "pred_class": torch.ones(1 + frame * 2, dtype=torch.long),
            "pos_key": (torch.arange(1 + frame * 2, dtype=dtype)).unsqueeze(1),
        }
        for frame in range(0, 10)
    ]

    trk = MultiStageTracker(
        fields=[
            TensorDictModule(nn.Identity(), in_keys=["pos_key"], out_keys=["pos"]),
            TensorDictModule(
                nn.Identity(), in_keys=["pred_class"], out_keys=["categories"]
            ),
        ],
        stages=[
            stages.Association(
                cost=costs.Distance("pos"), assignment=assignment.Jonker(10)
            )
        ],
    )

    mem = TrackletMemory(
        states={
            "pos": states.Value(dtype),
            "categories": states.Value(dtype=torch.long),
        }
    )

    for frame, det in enumerate(frames):
        det = TensorDict.from_dict(det)
        ctx, obs = mem.read(frame)
        obs, new = trk(ctx, obs, det)
        ids = mem.write(ctx, obs, new)

        assert len(ids) == len(det["pos_key"])
        assert torch.all(ids == torch.arange(len(det["pos_key"]), dtype=torch.long) + 1)
        assert isinstance(ids, torch.Tensor), type(ids)
