from abc import abstractmethod

import torch


class State(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reset()

    def forward(self, update: torch.Tensor, extend: torch.Tensor) -> None:
        """
        Set the current state from the given update and extend tensors. The update
        tensor is used to update the current state (i.e. propagate assigned objects), 
        while the extend tensor is used to extend the current state (i.e. add new objects).
        """

        self.update(update)
        self.extend(extend)

    @abstractmethod
    def update(self, update: torch.Tensor) -> None:
        """
        Update the current state from the given update tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def extend(self, extend: torch.Tensor) -> None:
        """
        Extend the current state from the given extend tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> torch.Tensor:
        """
        Observe the current (forecasted) state.

        Returns
        -------
            State value
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
