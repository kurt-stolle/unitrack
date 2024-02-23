from __future__ import annotations

from typing import Tuple

from tensordict import TensorDictBase

from ..assignment import Assignment
from ..constants import KEY_FRAME, KEY_ID, KEY_INDEX
from ..costs import Cost
from .base_stage import Stage

__all__ = ["Association"]


class Association(Stage):
    """
    An association stage assigns new track IDs using a cost matrix computed via
    a :class:`.Cost` module.
    """

    def __init__(
        self,
        cost: Cost,
        assignment: Assignment,
    ) -> None:
        super().__init__()

        self.cost = cost
        self.assignment = assignment

        self._extend_required_fields(self.cost)

    def _extend_required_fields(self, *costs: Cost) -> None:
        for cost in costs:
            self.required_fields += cost.required_fields

    def forward(
        self, ctx: TensorDictBase, cs: TensorDictBase, ds: TensorDictBase
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        if len(cs) == 0 or len(ds) == 0:
            return cs, ds

        cost_matrix = self.cost(cs, ds)

        matches, cs_fail_idx, ds_fail_idx = self.assignment(cost_matrix)

        if len(matches) > 0:
            cs_match = cs.get_sub_tensordict(matches[:, 0])
            ds_match = ds.get_sub_tensordict(matches[:, 1])

            self.match(cs_match, ds_match)

        cs_fail = cs.get_sub_tensordict(cs_fail_idx)
        ds_fail = ds.get_sub_tensordict(ds_fail_idx)

        return cs_fail, ds_fail

    def match(self, cs: TensorDictBase, ds: TensorDictBase) -> None:
        """
        Match candidates to detections. Propagates data and IDs from detections to candidates.

        Parameters
        ----------
        cs
            Candidates
        ds
            Detections
        """
        for key, value in ds.items():
            if key.startswith("_"):
                continue
            if key not in cs.keys():
                continue
            cs.set_(key, value)

        cs.set_(KEY_INDEX, ds.get(KEY_INDEX))
