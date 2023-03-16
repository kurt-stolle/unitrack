from typing import cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from unitrack import Detections, costs, stages


@pytest.mark.parametrize("jit", [False, True])
def test_lost_stage(jit):
    for num_frames in range(1, 10):
        for max_lost in range(1, num_frames):

            stage_lost = stages.Lost(max_lost=max_lost)
            if jit:
                stage_lost = torch.jit.script(stage_lost)

            ctx = stages.StageContext(
                frame=num_frames + 1,
                num_tracks=3,
                device=torch.device("cpu"),
                data={},
            )

            cs = Detections(
                {"_frame": torch.arange(num_frames, dtype=torch.long)}
            )
            ds = Detections({"_frame": torch.empty((0,), dtype=torch.long)})

            assert len(cs) == num_frames, cs
            assert len(ds) == 0, ds

            cs, ds = stage_lost(ctx, cs, ds)

            assert len(cs) == num_frames - max_lost, cs
            assert len(ds) == 0, ds
            assert torch.all(
                torch.arange(num_frames - max_lost) == cs._frame
            ), cs._frame


@pytest.mark.parametrize("jit", [False, True])
@settings(
    deadline=None,
)
@given(cs_num=st.integers(0, 10), ds_num=st.integers(0, 10))
def test_association_stage(cs_num: int, ds_num: int, jit):
    x_cs = cs_num * 2.0
    cs = Detections(
        {
            "x": torch.arange(cs_num).unsqueeze(1).float() * x_cs,
            "categories": torch.ones(cs_num),
        }
    )
    assert len(cs) == cs_num, cs
    x_ds = ds_num * 3.0 + 1
    ds = Detections(
        {
            "x": torch.arange(ds_num).unsqueeze(1).float() * x_ds,
            "categories": torch.ones(ds_num),
        }
    )
    assert len(ds) == ds_num, ds

    cost = costs.Distance(field="x")
    ass_stage = stages.Association(cost, torch.inf)

    if jit:
        ass_stage = cast(stages.Association, torch.jit.script(ass_stage))

    ass_num = min(cs_num, ds_num)

    ctx = stages.StageContext(
        frame=3, num_tracks=ds_num, device=torch.device("cpu"), data={}
    )
    cs, ds = ass_stage(ctx, cs, ds)

    assert len(cs) == cs_num - ass_num
    assert len(ds) == ds_num - ass_num

    if ass_num > 0:
        indices = ctx.matches[ctx.matches >= 0]
        assert len(indices) == ass_num
    else:
        assert len(ctx.matches[ctx.matches >= 0]) == 0
