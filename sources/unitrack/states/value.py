"""
Implements memory that stores a Tensor value.
"""

from __future__ import annotations

import typing as T
import typing_extensions as TX
import torch
import warnings

from .base_state import State

from unipercept.types import Size, Tensor, DType

DTypeSpec: T.TypeAlias = DType | str

DEFAULT_STATE_SLOTS = 1024


def _cast_dtype(dtype: DTypeSpec) -> torch.dtype:
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
    mask: Tensor

    def __init__(
        self,
        dtype: DTypeSpec = torch.float32,
        shape: Size | T.Iterable[int] = (),
        slots: int = DEFAULT_STATE_SLOTS,
    ):
        super().__init__()

        assert slots > 0, "Number of slots must be positive!"

        # Register the actual storage for values in memory
        dtype = _cast_dtype(dtype)
        shape = (slots, *shape)
        alloc = torch.zeros(shape, dtype=dtype, requires_grad=False)
        self.register_buffer("memory", alloc, persistent=False)

        # Register a mask to keep track of filled slots
        mask = torch.zeros(slots, dtype=torch.bool)
        self.register_buffer("mask", mask, persistent=False)

    @property
    @TX.override
    def slots(self) -> int:
        return self.mask.shape[0]

    @slots.setter
    @TX.override
    def slots(self, slots: int) -> None:
        max_alloc_index = torch.arange(slots, device=self.mask.device)[self.mask].max()
        if slots < max_alloc_index + 1:
            msg = f"Cannot set slots to {slots} as memory is already filled up to index {max_alloc_index}!"
            raise RuntimeError(msg)

        delta = slots - self.slots

        self.mask.resize_(slots)
        self.memory.resize_(slots, *self.shape)

        if delta > 0:
            self.mask[-delta:].zero_()
            self.memory[-delta:].zero_()

    @property
    @TX.override
    def dtype(self) -> torch.dtype:
        return self.memory.dtype

    @property
    @TX.override
    def shape(self) -> Size:
        return self.memory.shape[1:]

    @TX.override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.memory.dtype}, shape={self.memory.shape}){{{self.memory.tolist()}}}"

    @TX.override
    def read(self) -> dict[str, Tensor]:
        return {
            "memory": self.memory[self.mask],
        }

    @TX.override
    def update(self, memory: Tensor) -> None:
        self._check_compatible(memory)

        self.memory[self.mask, ...] = memory

    @TX.override
    def extend(self, memory: Tensor) -> None:
        self._check_compatible(memory)

        assert self.mask.device == self.memory.device == memory.device, (
            self.mask.device,
            self.memory.device,
            memory.device,
        )

        ext_len = memory.shape[0]
        if ext_len == 0:
            warnings.warn("Extending memory with empty tensor!")
            return

        free_slots = torch.arange(self.slots, device=self.mask.device)[~self.mask]
        if free_slots.shape[0] < ext_len:
            msg = f"Cannot extend memory by {ext_len} slots as only {free_slots.shape[0]} slots are available!"
            raise RuntimeError(msg)

        free_slots = free_slots[:ext_len]

        self.memory[free_slots] = memory
        self.mask[free_slots] = True

    @TX.override
    def observe(self) -> Tensor:
        mem = self.memory[self.mask, ...]
        assert mem.shape[1:] == self.shape, "Memory shape mismatch!"
        return mem

    @TX.override
    def reset(self) -> None:
        self.memory.zero_()
        self.mask.zero_()


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
