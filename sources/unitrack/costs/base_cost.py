from abc import abstractmethod
from typing import Iterable, Optional, Sequence

import torch
from tensordict import TensorDictBase

__all__ = ["Cost"]


class Cost(torch.nn.Module):
    """
    A cost module computes an assignment cost matrix between detections and
    tracklets.
    """

    required_fields: torch.jit.Final[list[str]]

    def __init__(self, required_fields: Iterable[str]):
        super().__init__()

        self.required_fields = list(set(required_fields))

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        return self.compute(cs, ds)

    @abstractmethod
    def compute(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        """
        Computes the assignment costs between :class:`.Tracklets` and
        :class:`.TensorDictBase`.

        This is an abstract method that should be overwritten be subclassses.

        Parameters
        ----------
        ts
            Tracklets instance.
        ds
            TensorDictBase instance.

        Returns
        -------
            Cost matrix
        """
        raise NotImplementedError
