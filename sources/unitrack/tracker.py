from __future__ import annotations

import typing as T

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

    def __init__(self, fields: T.Sequence[TensorDictModule], stages: T.Sequence[Stage]):
        super().__init__()

        assert len(stages) > 0

        self.fields = nn.ModuleList(fields)
        self.stages = nn.ModuleList(stages)

    def forward(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        inp: TensorDictBase,
        num: int,
    ) -> T.Tuple[TensorDictBase, TensorDictBase]:
        """
        Perform tracking, returns a tuple of updated observations and the field-values of new tracklets.


        Parameters
        ----------
        ctx: TensorDictBase
            The current state's context
        obs: TensorDictBase
            The current state's observations (i.e. when ``.observe()`` is called on
            each field in memory)
        inp: TensorDictBase
            The state of the next frame, from which new detections are gathered at
            every field.
        num: int
            The amount of detections made. Fields must allocate within a TensorDict
            that enforces ``batch_size=[num]``.


        Returns
        -------
        Tuple[TensorDict, TensorDict]
            Updated observations and field-values of new tracklets.
        """

        assert inp.device is not None

        # Create a dict of new tracklet candidates by passing the input state to
        # each field
        new = TensorDict(
            {KEY_INDEX: torch.arange(num, device=inp.device, dtype=torch.int64)},
            batch_size=[num],
            device=inp.device,
            _run_checks=False,
        )
        obs = obs.to(device=inp.device)

        for field in self.fields:
            field(inp, tensordict_out=new)

        assert obs.device is not None
        assert new.device is not None

        # Candidates for matching are all active observed tracklets
        active_mask = obs.get(KEY_ACTIVE)
        if active_mask.any():
            obs_candidates = obs._get_sub_tensordict(active_mask)
            for stage in self.stages:
                obs_candidates, new = stage(ctx, obs_candidates, new)
                if len(obs_candidates) == 0 or len(new) == 0:
                    break

            if len(obs_candidates) > 0 and obs_candidates.batch_size[0] > 0:
                # tensor: torch.Tensor = obs_candidates.get(KEY_ACTIVE)
                # tensor.fill_(False)
                # obs_candidates.fill_(KEY_ACTIVE, False)
                obs_candidates.set_(
                    KEY_ACTIVE,
                    torch.full(
                        obs_candidates.batch_size, False, device=obs_candidates.device
                    ),
                )

        return obs, new
