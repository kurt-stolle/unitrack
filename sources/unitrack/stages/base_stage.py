from abc import abstractmethod
from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

from ..detections import Detections

__all__ = ["Stage", "StageContext"]


@torch.jit.script
class StageContext:
    def __init__(
        self,
        frame: int,
        num_tracks: int,
        device: torch.device,
        data: dict[str, torch.Tensor],
    ):
        self.frame = frame
        self.matches = torch.full((num_tracks,), -1, dtype=torch.long, device=device)
        self.dt = 1

        if data is None:
            data = {}
        self.data = data

    def match(self, cs: Detections, ds: Detections) -> None:
        """
        Match candidates to detections.

        Parameters
        ----------
        cs
            Candidates
        ds
            Detections
        """
        self.matches[ds.indices] = cs.indices


class Stage(torch.nn.Module):
    """
    Base class for stages in a ::class::`..Tracker`.

    Inputs to a stage are in the context data, or as fields of the detections.
    """

    required_fields: List[str]
    required_data: List[str]

    def __init__(
        self,
        required_fields: List[str],
        required_data: List[str],
    ):
        super().__init__()

        self.required_fields: List[str] = required_fields
        self.required_data: List[str] = required_data

    @torch.jit.unused
    def __repr__(self) -> str:
        req = ", ".join(self.required_fields)
        return f"{type(self).__name__}(fields=[{req}])"

    def forward(
        self,
        ctx: StageContext,
        cs: Detections,
        ds: Detections,
    ) -> Tuple[Detections, Detections]:
        raise NotImplementedError
        return self.run(ctx, cs, ds)
