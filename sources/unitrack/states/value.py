"""
Implements memory that stores a Tensor value.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor

from .base_state import State

DataType: TypeAlias = torch.dtype | str


def _cast_dtype(dtype: DataType) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"No a valid data type: {dtype}!")
    return dtype


class Value(State):
    """
    A :class:`.State` that holds memory as-is without any updating logic
    implemented.
    """

    memory: Tensor

    def __init__(self, dtype: DataType):
        super().__init__()
        dtype = _cast_dtype(dtype)

        self.register_buffer(
            "memory", torch.empty(0, dtype=dtype, requires_grad=False), persistent=False
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.memory.dtype}, shape={self.memory.shape}){{{self.memory.tolist()}}}"

    def update(self, memory: Tensor) -> None:
        self.memory = memory.detach()

    def extend(self, memory: Tensor) -> None:
        memory_cat = torch.cat([self.memory, memory], dim=0)
        self.memory = memory_cat

    def observe(self) -> Tensor:
        return self.memory

    def reset(self) -> None:
        self.memory = self.memory[:0]


class Meanmemory(State):
    """
    A :class:`.State` that computes the mean of its memory over a given window
    size.
    """

    memory_history: Tensor

    def __init__(self, window: int, dtype: torch.dtype | str):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"No a valid data type: {dtype}!")

        self.window = window
        self.register_buffer(
            "memory_history", torch.zeros((0,), dtype=dtype), persistent=False
        )

    def forward(self, memory: Tensor) -> None:
        raise NotImplementedError

    def extend(self, memory: Tensor) -> None:
        memory = memory.unsqueeze(0)
        memory = memory.expand((self.memory_history.shape[0], *memory.shape))

        self.memory_history = torch.cat([self.memory_history, memory], dim=1)

    def observe(self) -> Tensor:
        return torch.mean(self.memory_history, dim=0)

    def reset(self) -> None:
        self.memory_history = self.memory_history[:0]
