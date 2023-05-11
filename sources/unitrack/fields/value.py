from typing import Dict, Optional

import torch

from .base_field import Field

__all__ = ["Value"]


class Value(Field):
    """
    A :class:`.Field` that copies its value directly from the input data.
    """

    def __init__(self, key: str):
        super().__init__(required_keys=[key], required_data=[])

        self.key = key

    @torch.jit.export
    def extract(self, kvmap: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return kvmap[self.key]
