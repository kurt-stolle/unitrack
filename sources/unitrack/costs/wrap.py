from typing import Final, List, Optional, Tuple

import torch
from torch import Tensor

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

    def get_field(self, cs: Detections, ds: Detections) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return (1.0 - _cosine_similarity(ts_field, ds_field)) ** self.alpha


def _cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """
    Manual computation of the cosine similarity.

    Based on: https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re  # noqa: E501

    ```
    csim(a,b) = dot(a, b) / (norm(a) * norm(b))
            = dot(a / norm(a), b / norm(b))
    ```

    Clamp with min(eps) for numerical stability. The final dot
    product is computed via transposed matrix multiplication (see `torch.mm`).
    """
    a_norm = _stable_norm(a)
    b_norm = _stable_norm(b)

    return torch.mm(a_norm, b_norm.T)


def _stable_norm(t: torch.Tensor, eps=9e-4):
    norm = torch.linalg.vector_norm(t, dim=1, keepdim=True)
    return t / norm.clamp(min=eps)
