import torch

from ..detections import Detections
from .base_cost import Cost

__all__ = ["CategoryGate"]


class CategoryGate(Cost):
    """
    Returns a matrix where each tracklet/detection pair that have equal
    categories are set to `True` and  that have different categories are set to
    `False`.
    """

    def __init__(self, field="categories"):
        super().__init__(required_fields=(field,))

        self.field: str = field

    def compute(self, cs: Detections, ds: Detections) -> torch.Tensor:
        cs_cats = cs.get(self.field)
        ds_cats = ds.get(self.field)

        gate_matrix = ds_cats[None, :] - cs_cats[:, None]

        return torch.where(gate_matrix == 0, 1.0, torch.inf)
