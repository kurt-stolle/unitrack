from typing import cast

import pytest
import torch

from unitrack import (
    Frame, MultiStageTracker, TrackletMemory, assignment, costs, fields, stages,
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

    tracker = MultiStageTracker(
        fields={
            "pos": fields.Value(key="pos_key"),
            "categories": fields.Value(key="pred_class"),
        },
        stages=[stages.Association(cost=costs.Distance("pos"), assignment=assignment.Jonker(10))],
    )

    tracks = TrackletMemory(
        states={
            "pos": states.Value(dtype),
            "categories": states.Value(dtype=torch.long),
        }
    )

    for frame, kvmap in enumerate(frames):
        ctx = Frame(kvmap, frame=frame)
        obs = tracks.read()

        obs, new = tracker(ctx, obs)
        ids = tracks.write(ctx, obs, new)

        assert len(ids) == len(kvmap["pos_key"])
        assert torch.all(ids == torch.arange(len(kvmap["pos_key"]), dtype=torch.long) + 1)
        assert isinstance(ids, torch.Tensor), type(ids)
