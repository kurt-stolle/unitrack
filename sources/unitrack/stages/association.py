from typing import Tuple

from ..assignment import Jonker
from ..costs import CategoryGate, Cost
from ..detections import Detections
from .base_stage import Stage, StageContext

__all__ = ["Association"]


class Association(Stage):
    """
    An association stage assigns new track IDs using a cost matrix computed via
    a :class:`.Cost` module.
    """

    def __init__(
        self,
        cost: Cost,
        threshold: float,
    ) -> None:
        super().__init__([], [])

        self.cost = cost
        self.category_gate = CategoryGate()
        self.assignment = Jonker(threshold=threshold)

        self._extend_required_fields(self.cost, self.category_gate)

    def _extend_required_fields(self, *costs: Cost) -> None:
        for cost in costs:
            self.required_fields += cost.required_fields

    def forward(self, ctx: StageContext, cs: Detections, ds: Detections) -> Tuple[Detections, Detections]:
        if len(cs) == 0 or len(ds) == 0:
            return cs, ds

        cost_matrix = self.cost(cs, ds)
        cost_matrix *= self.category_gate(cs, ds)

        matches, cs_fail, ds_fail = self.assignment(cost_matrix)

        if len(matches) > 0:
            # idx_t, idx_d = matches.T
            ctx.match(cs[matches[:, 0]], ds[matches[:, 1]])

        return cs[cs_fail], ds[ds_fail]
