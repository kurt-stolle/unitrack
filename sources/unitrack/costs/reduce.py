from enum import Enum
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .. import Detections
from .base_cost import Cost


class Reduction(Enum):
    """
    Reduction method for :class:`.Reduce` cost module.
    """

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"
    


class Reduce(Cost):
    """
    A weighted cost reduction module.
    """

    weights: torch.Tensor

    def __init__(
        self,
        costs: Sequence[Cost],
        weights: Optional[Sequence[float]] = None,
        reduction: Reduction | str = Reduction.SUM,
    ):
        super().__init__(required_fields=tuple(set().union(c.required_fields for c in costs)))

        if isinstance(reduction, str):
            reduction = Reduction(reduction)

        self.reduction = reduction
        self.costs = nn.ModuleList(costs)

        if weights is None:
            weights = [1.0] * len(costs)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def compute(self, cs: Detections, ds: Detections) -> torch.Tensor:
        costs = torch.stack([w * cost(cs, ds) for w, cost in zip(self.weights, self.costs)])

        return self._reduce(costs)

    def _reduce(self, costs: Tensor) -> Tensor:
        if self.reduction == Reduction.SUM:
            return costs.sum(dim=0)
        if self.reduction == Reduction.MEAN:
            return costs.mean(dim=0)
        if self.reduction == Reduction.MIN:
            return costs.min(dim=0).values
        if self.reduction == Reduction.MAX:
            return costs.max(dim=0).values
        raise ValueError(f"Unknown reduction: {self.reduction}!")
