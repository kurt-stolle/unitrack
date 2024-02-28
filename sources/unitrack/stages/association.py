from __future__ import annotations

import typing as T

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
    ) -> T.Tuple[TensorDictBase, TensorDictBase]:
        if len(cs) == 0 or len(ds) == 0:
            return cs, ds

        cost_matrix = self.cost(cs, ds)

        matches, cs_fail_idx, ds_fail_idx = self.assignment(cost_matrix)

        print(f"cs device: {cs.device}")
        print(f"ds device: {ds.device}")
        print(f"cost amatrix device: {cost_matrix.device}")
        print(f"matches device: {matches.device}")

        if len(matches) > 0:
            cs_match = cs._get_sub_tensordict(matches[:, 0])
            ds_match = ds._get_sub_tensordict(matches[:, 1])

            self.match(cs_match, ds_match)

        cs_fail = cs._get_sub_tensordict(cs_fail_idx)
        ds_fail = ds._get_sub_tensordict(ds_fail_idx)

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
        cs_keys = T.cast(T.List[str], cs.keys())
        ds_keys = T.cast(T.List[str], ds.keys())
        for key in ds_keys:
            if key.startswith("_"):
                continue
            if key not in cs_keys:
                continue
            cs.set_(key, ds.get(key))
        # pass the index of the detection such that it can be retrieved later when
        # collecting the retuned ID
        cs.set_(KEY_INDEX, ds.get(KEY_INDEX))
