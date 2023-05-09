from abc import abstractmethod
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from ..detections import Detections

__all__ = ["Cost", "WeightedReduce"]


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
                raise ValueError(f"Missing field '{field_name}' in trackets: {cs}!")
            if field_name not in ds:
                raise ValueError(f"Missing field '{field_name}' in detections: {cs}!")

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

