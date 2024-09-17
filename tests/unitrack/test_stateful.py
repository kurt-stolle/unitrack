r"""
Tests for ``unitrack.stateful``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import unitrack as ut

@pytest.fixture
def stateful_tracker():
    return ut.StatefulTracker(
        tracker=ut.MultiStageTracker(
            fields=[
                _build_field(TRACK_KEY_LABEL, ("panoptic_ids")),
                _build_field(TRACK_KEY_SCORE, ("scores")),
                _build_field(TRACK_KEY_CATEGORY, ("categories")),
                TensorDictModule(
                    module=MaskToBoxes(common_stride),
                    in_keys=[
                        ("panoptic_ids"),
                        ("outputs", OUT_MAPPING),
                    ],
                    out_keys=[TRACK_KEY_BBOX],
                ),
            ],
            stages=[
                ut.stages.Gate(
                    gate=SelectAndFilter(s),
                    then=[
                        ut.stages.Association(
                            cost=cost,
                            assignment=ut.assignment.Jonker(threshold=threshold),
                        ),
                    ],
                )
                for s in [0.1]
            ],
        ),
        memory=ut.TrackletMemory(
            states={
                TRACK_KEY_SCORE: ut.states.Value(torch.float),
                TRACK_KEY_CATEGORY: ut.states.Value(torch.long),
                TRACK_KEY_BBOX: ut.states.Value(torch.float),
            },
        ),
    )


def test_stateful_tracker():
    
