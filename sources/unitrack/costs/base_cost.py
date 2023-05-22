from abc import abstractmethod
from typing import Iterable, Optional, Sequence

import torch
from tensordict import TensorDictBase

__all__ = ["Cost", "FieldCost"]


class Cost(torch.nn.Module):
    """
    A cost module computes an assignment cost matrix between detections and
    tracklets.
    """

    required_fields: torch.jit.Final[list[str]]

    def __init__(self, required_fields: Iterable[str]):
        super().__init__()

        self.required_fields = list(set(required_fields))

    @abstractmethod
    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        """
        Computes the assignment costs between previous tracklets and current
        detections.

        This is an abstract method that should be overwritten.

        Parameters
        ----------
        cs
            Candidate observations (N).
        ds
            Detections (M).
        Returns
        -------
            Cost matrix (N x M).
        """
        raise NotImplementedError


class FieldCost(Cost):
    field: torch.jit.Final[str]

    def __init__(self, field: str):
        super().__init__(required_fields=[field])

        self.field = field

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        return self.compute(cs.get(self.field), ds.get(self.field))

    @abstractmethod
    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        """
        Computes the assignment costs between previous tracklets and current
        detections.

        This is an abstract method that should be overwritten.

        Parameters
        ----------
        cs
            Candidate observations (N) values for a single field.
        ds
            Detections (M) values for a single field.

        Returns
        -------
            Cost matrix (N x M).
        """
        raise NotImplementedError
