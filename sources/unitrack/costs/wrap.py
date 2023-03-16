from typing import Final, List, Optional, Tuple

import torch

from .....utils.cosine import cosine_distance
from ..detections import Detections
from .base_cost import Cost


class Distance(Cost):
    """
    Cost function that wraps a distance module.
    """

    select: List[int]
    p: float

    def __init__(
        self,
        field: str,
        select: Optional[List[int]] = None,
        p_norm: float = 2.0,
        alpha: float = 1.0,
    ):
        """
        Parameters
        ----------
        critereon
            Module that computes distance between two fields
        field_name
            Name if the field to read tensors from that are passed to the
            ``torch.cdist`` function.
        select
            List of indices to select from the field.
        p_norm
            Value of p-norm.
        alpha
            Result distance ^ alpha
        """
        super().__init__(required_fields=(field,))

        self.field = field

        if select is None:
            select = []

        self.select = select  # type: ignore
        self.p_norm = p_norm
        self.alpha = alpha

    def get_field(
        self, cs: Detections, ds: Detections
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ts_field = cs.get(self.field)
        ds_field = ds.get(self.field)

        if len(self.select) > 0:
            assert ts_field.shape[1] >= max(self.select)
            assert ds_field.shape[1] >= max(self.select)

            ts_field = ts_field[:, self.select]
            ds_field = ds_field[:, self.select]

        return ts_field, ds_field

    def compute(self, cs: Detections, ds: Detections):
        ts_field, ds_field = self.get_field(cs, ds)
        return torch.cdist(ts_field, ds_field, self.p_norm) * self.alpha


class Cosine(Distance):
    """
    Computes the distance between two fields based on a similarity measure with
    range $[0,1]$ by computing $1 - S(x,y)$.

    For example, a wrapped module could be ``torch.nn.CosineSimilarity``.
    """

    def compute(self, cs: Detections, ds: Detections):
        ts_field, ds_field = self.get_field(cs, ds)
        return cosine_distance(ts_field, ds_field) ** self.alpha
