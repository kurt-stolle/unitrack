from typing import cast

import pytest
import torch
from tensordict import TensorDict

from unitrack import (
    MultiStageTracker, TrackletMemory, assignment, costs, fields, stages,
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
        fields={
            "pos": fields.Value(key="pos_key"),
            "categories": fields.Value(key="pred_class"),
        },
        stages=[stages.Association(cost=costs.Distance("pos"), assignment=assignment.Jonker(10))],
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

        assert len(ids) == len(obs["pos_key"])
        assert torch.all(ids == torch.arange(len(obs["pos_key"]), dtype=torch.long) + 1)
        assert isinstance(ids, torch.Tensor), type(ids)
