from __future__ import annotations

import copy
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from . import MultiStageTracker, TrackletMemory

__all__ = ["StatefulTracker"]


class _MemoryReadWriter(nn.Module):
    """
    Wrapper around a tracklet memory that enables writing and reading in the forward pass.
    This is necessary because the memory is a stateful module that is not compatible with the functional API.
    """

    def __init__(self, memory: TrackletMemory):
        super().__init__()
        self.memory = memory

    @override
    def forward(
        self,
        write: bool,
        transaction: int | tuple[TensorDictBase, TensorDictBase, TensorDictBase],
    ):
        if write:
            ctx, obs, new = transaction
            return self.memory.write(ctx, obs, new)
        else:
            (frame,) = transaction
            return self.memory.read(frame)


def _split_persistent_buffers(
    module: nn.Module, prefix: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Split the buffers of a module into two dictionaries: one for
    buffers that are shared across sequences (persistent) and one for buffers that are
    unique to every sequence (non-persistent).
    """

    shared = {}
    unique = {}

    for name, buf in module.named_buffers(prefix="", recurse=False):
        name_prefixed = f"{prefix}.{name}"
        if name in module._non_persistent_buffers_set:
            unique[name_prefixed] = buf
        else:
            shared[name_prefixed] = buf

    for name, submodule in module.named_children():
        s, u = _split_persistent_buffers(submodule, prefix=f"{prefix}.{name}")
        shared.update(s)
        unique.update(u)

    return shared, unique


class StatefulTracker(nn.Module):
    """
    A wrapper around tracker and tracklets that enables
    stateful tracking of objects for every separate sequence.
    """

    mem_buffers: Mapping[str | int, dict[str, torch.Tensor]]
    mem_params: dict[str, torch.Tensor]

    def __init__(self, tracker: MultiStageTracker, memory: TrackletMemory):
        super().__init__()

        self.tracker = tracker
        self.memory_delegate = _MemoryReadWriter(memory)

    @cached_property
    def memory_storage(
        self,
    ) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.Tensor], defaultdict]:
        prefix = "memory"
        memory = self.memory_delegate.memory
        params = dict(memory.named_parameters(prefix=prefix))
        buffers_shared, buffers_unique = _split_persistent_buffers(
            memory, prefix=prefix
        )
        buffers_unique_map = defaultdict(lambda: copy.deepcopy(buffers_unique))

        return params, buffers_shared, buffers_unique_map

    @override
    def forward(self, x: TensorDict, n: int, key: int, frame: int) -> Tensor:
        """
        Parameters
        ----------
        x: TensorDictBase
            Represents the state of the current iteration.
        n: int
            The amount of detections in the current frame, amounts to the length of
            IDs returned
        key: int
            The key that identifies the sequence in which the detections have been made
        frame: int
            The current frame number, used to identify the temporal position within
            the sequence.

        Returns
        -------
        Tensor[n]
            Assigned instance IDs
        """
        # Read
        params, buffers_shared, buffers_unique = self.memory_storage
        buffers_unique = buffers_unique[key]
        pbd: dict[str, torch.Tensor] = {**params, **buffers_shared, **buffers_unique}
        state_ctx, state_obs = torch.func.functional_call(
            self.memory_delegate, pbd, (False, (frame,)), strict=True
        )

        # Step
        state_obs, new = self.tracker(state_ctx, state_obs, x, n)

        # Write
        ids: torch.Tensor = torch.func.functional_call(
            self.memory_delegate, pbd, (True, (state_ctx, state_obs, new)), strict=True
        )

        for buf_key, buf_val in pbd.items():
            if buf_key in params:
                continue
            if buf_key in buffers_shared:
                buffers_shared[buf_key] = buf_val
                continue
            if buf_key in buffers_unique:
                buffers_unique[buf_key] = buf_val
                continue

            msg = (
                f"Buffer with key {buf_key!r} not found in parameters and buffers dict!"
            )
            raise KeyError(msg)

        if len(ids) != n:
            msg = f"Expected {n} IDs to be returned, but got {len(ids)} instead!"
            raise ValueError(msg)

        return ids
