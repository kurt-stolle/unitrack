from typing import Dict, Optional

import torch

from .base_field import Field

__all__ = ["Value"]


class Value(Field):
    """
    A :class:`.Field` that copies its value directly from the input data.
    """

    def __init__(self, id: str, key: Optional[str] = None):
        if key is None:
            key = id

        super().__init__(id, required_keys=[key], required_data=[])

        self.key = key

    @torch.jit.export
    def extract(self, kvmap: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return kvmap[self.key]
