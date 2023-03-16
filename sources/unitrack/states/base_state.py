from abc import abstractmethod
from typing import Any

import torch


class State(torch.nn.Module):
    id: str

    def __init__(self, id: str):
        super().__init__()

        self.id = id

    @torch.jit.ignore
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id})"

    @torch.jit.export
    def update(self, value: Any) -> None:
        """
        Set the current state from new values.

        Parameters
        ----------
        value
            Value to update current state with.

        Returns
        -------
            New state.
        """
        raise NotImplementedError

    @torch.jit.export
    def extend(self, value: Any) -> None:
        """
        Add new values to the state.

        Parameters
        ----------
        value
            Values to add

        Returns
        -------
            New state

        """
        raise NotImplementedError

    @torch.jit.export
    def observe() -> torch.Tensor:
        """
        Observe the current (predicted) state.

        Returns
        -------
            State value
        """
        raise NotImplementedError

    @torch.jit.export
    def reset(self):
        raise NotImplementedError
