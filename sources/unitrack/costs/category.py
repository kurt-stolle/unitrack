import torch
from tensordict import TensorDictBase

from .base_cost import FieldCost

__all__ = ["CategoryGate"]


class CategoryGate(FieldCost):
    """
    Returns a matrix where each tracklet/detection pair that have equal
    categories are set to `True` and  that have different categories are set to
    `False`.
    """

    def __init__(self, field="categories"):
        super().__init__(field=field)

    def compute(self, cs_cats: torch.Tensor, ds_cats: torch.Tensor) -> torch.Tensor:
        gate_matrix = ds_cats[None, :] - cs_cats[:, None]

        return gate_matrix == 0
