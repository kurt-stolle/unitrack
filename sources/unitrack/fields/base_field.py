from abc import abstractmethod
from typing import Dict, List

import torch

from ..context import Context

from tensordict import TensorDict

__all__ = ["Field"]


class Field(torch.nn.Module):
    """
    A field is a module that transforms data into state items of a tracked
    object.
    """

    def __init__(self, required_keys: List[str], required_data: List[str]):
        """
        Parameters
        ----------
        id
            Unique ID for this field.
        required_data
            List of required keys in the input data.
        """

        super().__init__()

        self.required_keys = required_keys
        self.required_data = required_data

    def forward(self, ctx: Context):
        return self.extract(ctx)

    @torch.jit.unused
    def __repr__(self) -> str:
        req = ", ".join(self.required_keys)
        return f"{type(self).__name__}({self.id}, " f"data=[{req}])"

    @abstractmethod
    def extract(self, ctx: Context) -> torch.Tensor | TensorDict:
        """
        Extract field values from data.

        Parameters
        ----------
        data
            Mapping of data keys to tensors.

        Returns
        -------
            Mapping of field names to values.
        """
        raise NotImplementedError
