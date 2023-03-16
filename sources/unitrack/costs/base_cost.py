from abc import abstractmethod
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from ..detections import Detections


class Cost(nn.Module):
    """
    A cost module computes an assignment cost matrix between detections and
    tracklets.
    """

    def __init__(self, required_fields: tuple[str]):
        super().__init__()

        self.required_fields = required_fields

    def forward(self, cs: Detections, ds: Detections) -> Tensor:
        for field_name in self.required_fields:
            if field_name not in cs:
                raise ValueError(
                    f"Missing field '{field_name}' in trackets: {cs}!"
                )
            if field_name not in ds:
                raise ValueError(
                    f"Missing field '{field_name}' in detections: {cs}!"
                )

        return self.compute(cs, ds)

    @abstractmethod
    def compute(self, cs: Detections, ds: Detections) -> Tensor:
        """
        Computes the assignment costs between :class:`.Tracklets` and
        :class:`.Detections`.

        This is an abstract method that should be overwritten be subclassses.

        Parameters
        ----------
        ts
            Tracklets instance.
        ds
            Detections instance.

        Returns
        -------
            Cost matrix
        """
        raise NotImplementedError


class WeightedReduce(Cost):
    """
    A weighted reduction module.
    """

    weights: Tensor

    def __init__(
        self,
        costs: Sequence[Cost],
        weights: Optional[Sequence[float]] = None,
        reduction="sum",
    ):
        super().__init__(
            required_fields=tuple(set().union(c.required_fields for c in costs))
        )

        self.reduction = reduction
        self.costs = nn.ModuleList(costs)
        if weights is None:
            weights = [1.0] * len(costs)

        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float)
        )

    def compute(self, cs: Detections, ds: Detections) -> Tensor:
        costs = torch.stack(
            [w * cost(cs, ds) for w, cost in zip(self.weights, self.costs)]
        )

        if self.reduction == "sum":
            return costs.sum(dim=0)
        if self.reduction == "mean":
            return costs.mean(dim=0)
        if self.reduction == "min":
            return costs.min(dim=0).values
        if self.reduction == "max":
            return costs.max(dim=0).values
        raise ValueError(f"Unknown reduction: {self.reduction}!")
