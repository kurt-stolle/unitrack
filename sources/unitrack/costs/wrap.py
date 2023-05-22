from typing import Final, List, Optional, Tuple

import torch
from tensordict import TensorDictBase
from torch import Tensor

from .base_cost import Cost

__all__ = ["Distance", "Cosine"]


DEFAULT_EPS: Final = 1e-7


class Distance(Cost):
    """
    Cost function that wraps a distance module.
    """

    select: torch.jit.Final[List[int]]
    p: torch.jit.Final[float]
    field: torch.jit.Final[str]

    def __init__(
        self,
        field: str,
        select: Optional[List[int]] = None,
        p_norm: float = 2.0,
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
        """
        super().__init__(required_fields=(field,))

        self.field = field

        if select is None:
            select = []

        self.select = select  # type: ignore
        self.p = p_norm

    def get_field(self, cs: TensorDictBase, ds: TensorDictBase) -> Tuple[torch.Tensor, torch.Tensor]:
        ts_field = cs.get(self.field)
        ds_field = ds.get(self.field)

        if len(self.select) > 0:
            assert ts_field.shape[1] >= max(self.select)
            assert ds_field.shape[1] >= max(self.select)

            ts_field = ts_field[:, self.select]
            ds_field = ds_field[:, self.select]

        return ts_field, ds_field

    def forward(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return torch.cdist(ts_field, ds_field, self.p, "donot_use_mm_for_euclid_dist")


class Cosine(Distance):
    """
    Computes the distance between two fields based on a similarity measure with
    range $[0,1]$ by computing $1 - S(x,y)$.
    """

    def __init__(self, *args, eps: float = DEFAULT_EPS, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def compute(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return 1.0 - _cosine_similarity(ts_field, ds_field, eps=self.eps)


def _cosine_similarity(a: Tensor, b: Tensor, eps=DEFAULT_EPS) -> Tensor:
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
    a_norm = _stable_norm(a, eps=eps)
    b_norm = _stable_norm(b, eps=eps)

    return torch.mm(a_norm, b_norm.T)


def _stable_norm(t: torch.Tensor, eps=DEFAULT_EPS):
    norm = torch.linalg.vector_norm(t, dim=1, keepdim=True)
    return t / norm.clamp(min=eps)
