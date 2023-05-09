from typing import Tuple

import torch

from ..structures import Detections
from .base_stage import Stage, StageContext

__all__ = ["Lost"]


class Lost(Stage):
    """
    Stage that filters out candidates that have been lost for less than a
    configurable maximum amount.


    The time lost is computed via $$ T_l = t - T_c - dt $$ where $t$ is the
    current time (i.e. frame) and $T_c$ are the candidate time states.
    The value is corrected with time-step $dt$ to account for candidates not yet
    being updated to the current time, as the state update is performed
    *after* all stages have been ran.
    """

    max_lost: torch.jit.Final[int]

    def __init__(self, max_lost: int):
        """
        Parameters
        ----------
        max_lost
            Maximum amount of time a candidate may remain lost.
        """

        super().__init__([], [])

        assert max_lost > 0, max_lost

        self.max_lost = max_lost

    def forward(self, ctx: StageContext, cs: Detections, ds: Detections) -> Tuple[Detections, Detections]:
        if len(cs) == 0:
            return cs, ds

        time_lost = ctx.frame - cs.get("_frame") - ctx.dt

        return cs[time_lost > self.max_lost], ds
