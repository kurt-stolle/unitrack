from abc import abstractmethod
from typing import Iterable, TypeAlias, cast

import torch
from torch import Tensor

__all__ = ["State"]

StateValue: TypeAlias = Tensor | dict[str, Tensor]


class State(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def patch(self, update: StateValue) -> None:
        """
        Update the current state from the given update tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def put(self, extend: StateValue) -> None:
        """
        Extend the current state from the given extend tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> StateValue:
        """
        Observe the current state.

        Returns
        -------
            State value
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset the state to its initial value. Essentially sets the buffers to their initial value.
        """

        raise NotImplementedError

    def read(self) -> list[tuple[str, Tensor]]:
        """
        Read the current state.

        Returns
        -------
            State value
        """
        items = cast(list[tuple[str, Tensor]], list(self._buffers.items()))
        assert len(items) > 0, "No buffers found!"

        return items

    def evolve(self, delta: Tensor) -> Iterable[tuple[str, Tensor]]:
        """
        Evolve the state by the given delta. This is a no-op by default.

        Parameters
        ----------
        delta
            The time difference to evolve the state by.
        """
        return self.observe()
