from tensordict import TensorDictBase
from torch import Tensor

from .base_field import Field

__all__ = ["Value"]


class Value(Field):
    """
    A :class:`.Field` that copies its value directly from the input data.
    """

    def __init__(self, key: str, **kwargs):
        super().__init__(in_det=[key], in_ctx=[], **kwargs)

    def forward(self, value: Tensor) -> Tensor:
        return value
