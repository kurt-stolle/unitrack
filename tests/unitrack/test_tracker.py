from typing import cast

import pytest
import torch

from unitrack import Tracker, Tracklets, costs, fields, stages, states
from unitrack.tracker import TrackerResult


@pytest.mark.parametrize("jit", [False, True])
def test_tracker(jit):
    dtype = torch.float32
    frames = [
        {
            "pred_class": torch.ones(1 + frame * 2, dtype=torch.long),
            "pos_key": (torch.arange(1 + frame * 2, dtype=dtype)).unsqueeze(1),
        }
        for frame in range(0, 10)
    ]

    tracker = Tracker(
        fields=[
            fields.Value(id="pos", key="pos_key"),
            fields.Value(id="categories", key="pred_class"),
        ],
        stages=[
            stages.Association(cost=costs.Distance("pos"), threshold=999999)
        ],
    )
    if jit:
        tracker = cast(Tracker, torch.jit.script(tracker))  # type: ignore

    tracks = Tracklets(
        states=[
            states.Value(id="pos", dtype=dtype),
            states.Value(id="categories", dtype=torch.long),
        ]
    )
    if jit:
        tracks = cast(Tracklets, torch.jit.script(tracks))  # type: ignore

    for frame, kvmap in enumerate(frames):
        state_obs = tracks.observe()

        res = tracker(frame, state_obs, kvmap, {})
        assert len(res.matches) == len(kvmap["pos_key"])

        ids = tracks("seq", frame, res)

        assert len(ids) == len(kvmap["pos_key"])
        assert torch.all(
            ids == torch.arange(len(kvmap["pos_key"]), dtype=torch.long) + 1
        )
        assert isinstance(ids, torch.Tensor), type(ids)
