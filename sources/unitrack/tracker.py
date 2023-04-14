from typing import Dict, List, NamedTuple, Optional
from warnings import warn

import torch
from torch import Tensor, nn

from .detections import Detections
from .fields import Field
from .stages import Stage, StageContext

__all__ = ["Tracker"]


class TrackerResult(NamedTuple):
    update: Detections
    extend: Detections
    matches: Tensor


class Tracker(nn.Module):
    """
    Tracker is a module that transforms detections into a :class:`.Tracklets`
    module.
    """

    def __init__(self, fields: List[Field], stages: List[Stage]):
        super().__init__()

        assert len(stages) > 0

        self.stages = nn.ModuleList(stages)
        self.fields = nn.ModuleDict({field.id: field for field in fields})

        self._validate()

    def _validate(self):
        for i, stage in enumerate(self.stages):
            if not isinstance(stage, Stage):
                warn(f"Stage {i} is not an instance of `Stage`.")
                continue
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
        frame: int,
        states: Detections,
        kvmap: dict[str, Tensor],
        data: dict[str, Tensor],
    ) -> TrackerResult:
        assert frame >= 0, frame

        # Create detections through each field
        detections = self._apply_fields(kvmap, data)

        # Create context
        ctx = StageContext(
            frame=frame,
            num_tracks=len(detections),
            device=detections.indices.device,
            data=data,
        )

        # Apply stage cascade
        return self._apply_stages(ctx, states, detections)

    def _apply_fields(self, kvmap: dict[str, Tensor], data: dict[str, Tensor]) -> Detections:
        items = {key: field(kvmap, data) for key, field in self.fields.items()}
        return Detections(items)  # type: ignore

    def _apply_stages(self, ctx: StageContext, states: Detections, detections: Detections) -> TrackerResult:
        # Sanity check
        obs_frames = states.get("_start")
        if not torch.all(obs_frames < ctx.frame):
            frame_list: List[int] = obs_frames.tolist()
            raise ValueError(f"Attempted to track at frame {ctx.frame} while `Tracklets` " f"has frames {frame_list}")

        # Select candidates based on active flag
        cs = states[states.get("_active")]
        ds = detections
        assert not cs.mutable
        for stage in self.stages:
            cs, ds = stage(ctx, cs, ds)

        # Handle remaining candidates
        if len(cs) > 0:
            # Update state to inactive
            active = states.get("_active").clone()
            # active[cs.indices] = torch.full_like(
            #     cs.indices, False, dtype=torch.bool
            # )
            active[cs.indices] = False

            states.set("_active", active)

        # Update matched detections -> candidates
        if ctx.matches.sum() > 0:
            obs_indices = ctx.matches[ctx.matches >= 0]
            ds_matched = detections[ctx.matches >= 0]

            # Update states that match a field name
            for key in self.fields.keys():
                if key in states:
                    measurements = states.get(key).clone()
                    measurements[obs_indices] = ds_matched.get(key)
                    states.set(key, measurements)

            # Update frame number
            frames = states.get("_frame").clone()
            frames[obs_indices].fill_(ctx.frame)

            states.set("_frame", frames)

        return TrackerResult(update=states, extend=ds, matches=ctx.matches)
