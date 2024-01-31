from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple, cast

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential

from .constants import KEY_ACTIVE, KEY_FRAME, KEY_ID, KEY_INDEX, KEY_START
from .stages import Stage


class MultiStageTracker(nn.Module):
    """
    Multi-stage tracker that applies a cascade of stages to a set of detections.
    """

    def __init__(self, fields: Sequence[TensorDictModule], stages: Sequence[Stage]):
        super().__init__()

        assert len(stages) > 0

        self.fields = nn.ModuleList(fields)
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

        # Fields target (newly detected objects)
        new = TensorDict(
            {},
            batch_size=[],
            device=inp.device,
        )

        for field in self.fields:
            field(inp, tensordict_out=new)

        # Infer the number of detections from the batch size of some element in the new detections
        num_det = int(next(iter(new.values())).shape[0])
        new.batch_size = torch.Size((num_det,))
        if num_det == 0:
            return obs, new

        # Add the index of the detections to association
        new[KEY_INDEX] = torch.arange(num_det, device=inp.device)

        # Candidates for matching are all active observed tracklets
        obs_active = obs.get(KEY_ACTIVE)
        if obs_active.any():
            obs_candidates = obs.get_sub_tensordict(obs_active)
            for stage in self.stages:
                obs_candidates, new = stage(ctx, obs_candidates, new)
                if len(obs_candidates) == 0 or len(new) == 0:
                    break

            if len(obs_candidates) > 0 and obs_candidates.batch_size[0] > 0:
                obs_candidates.fill_(KEY_ACTIVE, False)

        return obs, new
