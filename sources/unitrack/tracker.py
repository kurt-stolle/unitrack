from typing import Dict, List, Mapping, Tuple, cast

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from .constants import KEY_ACTIVE, KEY_FRAME, KEY_ID, KEY_INDEX, KEY_START
from .context import Context
from .fields import Field
from .stages import Stage


class MultiStageTracker(nn.Module):
    """
    Multi-stage tracker that applies a cascade of stages to a set of detections.
    """

    def __init__(self, fields: Mapping[str, Field], stages: List[Stage]):
        super().__init__()

        assert len(stages) > 0

        self.stages = cast(List[Stage], nn.ModuleList(stages))
        self.fields = cast(Dict[str, Field], nn.ModuleDict({**fields}))

        self._validate()

    def _validate(self):
        for i, stage in enumerate(self.stages):
            for id in stage.required_fields:
                if id not in self.fields:
                    raise ValueError(f"Field with ID '{id}' missing for {stage}!")

    def get_required_fields(self) -> List[str]:
        field_keys: List[str] = []
        for f in self.fields.values():
            field_keys += f.required_keys
        return field_keys

    def get_required_data(self) -> List[str]:
        data_keys: List[str] = []
        for f in self.stages:
            data_keys += f.required_data
        return data_keys

    def forward(
        self,
        ctx: Context,
        obs: TensorDictBase,
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        """
        Perform tracking, returns a tuple of updated observations and the field-values of new tracklets.

        Parameters
        ----------
        ctx
            Context object
        obs
            Observations
        det
            Detections

        Returns
        -------
        Tuple[TensorDict, TensorDict]
            Updated observations and field-values of new tracklets.
        """

        new = self._apply_fields(ctx)
        obs, new = self._apply_stages(ctx, obs, new)

        return obs, new

    def _apply_fields(self, ctx: Context) -> TensorDictBase:
        items = {key: field(ctx) for key, field in self.fields.items()}
        items[KEY_INDEX] = torch.arange(len(ctx.ids), device=ctx.ids.device)
        return TensorDict(items, batch_size=ctx.ids.shape, device=ctx.ids.device)  # type: ignore

    def _apply_stages(
        self, ctx: Context, obs: TensorDictBase, new: TensorDictBase
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        obs_candidates = obs.get_sub_tensordict(obs.get(KEY_ACTIVE))
        for stage in self.stages:
            obs_candidates, new = stage(ctx, obs_candidates, new)

        if len(obs_candidates) > 0 and obs_candidates.batch_size[0] > 0:
            obs_candidates.fill_(KEY_ACTIVE, False)

        return obs, new
