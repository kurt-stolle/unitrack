import enum
import itertools
from enum import Enum
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from torch import Tensor

from .base_cost import Cost

__all__ = ["Reduce", "Reduction"]


class Weighted(Cost):
    """
    A weighted cost module.
    """

    def __init__(self, cost: Cost, weight: float):
        super().__init__(required_fields=cost.required_fields)

        self.cost = cost
        self.weight = weight

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        return self.weight * self.cost(cs, ds)


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
    A cost reduction module.
    """

    weights: torch.Tensor
    method: torch.jit.Final[Reduction]

    def __init__(
        self,
        costs: Sequence[Cost],
        method: Reduction,
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__(required_fields=itertools.chain(*(c.required_fields for c in costs)))

        if isinstance(method, str):
            method = Reduction(method)

        self.method = method
        self.costs = nn.ModuleList(costs)
        if weights is None:
            weights = [1.0] * len(costs)

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).unsqueeze_(-1).unsqueeze_(-1))

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        costs = torch.stack([cost(cs, ds) for cost in self.costs])
        costs = costs * self.weights
        costs = _reduce_stack(costs, self.method)

        return costs


@torch.jit.script_if_tracing
def _reduce_stack(costs: Tensor, method: Reduction) -> Tensor:
    if method == Reduction.SUM:
        return costs.sum(dim=0)
    if method == Reduction.MEAN:
        return costs.mean(dim=0)
    if method == Reduction.MIN:
        return costs.min(dim=0).values
    if method == Reduction.MAX:
        return costs.max(dim=0).values
    raise NotImplementedError(f"Reduction method '{method}' not implemented!")
