from typing import Any

import torch


class State(torch.nn.Module):
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
    def observe(self) -> torch.Tensor:
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
