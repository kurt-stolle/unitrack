from typing import Dict, List, Mapping, Sequence, Tuple, cast

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    TensorDictModule, TensorDictModuleBase, TensorDictSequential,
)

from .constants import KEY_ACTIVE, KEY_FRAME, KEY_ID, KEY_INDEX, KEY_START
from .stages import Stage


class MultiStageTracker(nn.Module):
    """
    Multi-stage tracker that applies a cascade of stages to a set of detections.
    """

    def __init__(self, fields: Sequence[TensorDictModule], stages: Sequence[Stage]):
        super().__init__()

        assert len(stages) > 0

        self.fields = TensorDictSequential(*fields)
        self.stages = cast(List[Stage], nn.ModuleList(stages))

    def forward(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        inp: TensorDictBase,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """
        Perform tracking, returns a tuple of updated observations and the field-values of new tracklets.


        Returns
        -------
        Tuple[TensorDict, TensorDict]
            Updated observations and field-values of new tracklets.
        """

        # Read the amount of detections
        num_det = inp.batch_size[0]

        # Fields target (newly detected objects)
        new = TensorDict(
            {
                KEY_INDEX: torch.arange(num_det, device=inp.device),
            },
            batch_size=[num_det],
            device=inp.device,
        )
        self.fields(inp, tensordict_out=new)

        obs_candidates = obs.get_sub_tensordict(obs.get(KEY_ACTIVE))
        for stage in self.stages:
            obs_candidates, new = stage(ctx, obs_candidates, new)

        if len(obs_candidates) > 0 and obs_candidates.batch_size[0] > 0:
            obs_candidates.fill_(KEY_ACTIVE, False)

        return obs, new
