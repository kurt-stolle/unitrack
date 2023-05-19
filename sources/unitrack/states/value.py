import torch

from .base_state import State


class Value(State):
    """
    A :class:`.State` that holds values as-is without any updating logic
    implemented.
    """

    values: torch.Tensor

    def __init__(self, dtype: torch.dtype | str):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"No a valid data type: {dtype}!")

        self.register_buffer("values", torch.empty(0, dtype=dtype), persistent=False)

    def forward(self, values: torch.Tensor) -> None:
        self.values = values

    def extend(self, values: torch.Tensor) -> None:
        values_cat = torch.cat([self.values, values], dim=0)
        self.values = values_cat

    def observe(self) -> torch.Tensor:
        return self.values

    def reset(self) -> None:
        self.values = self.values[:0]


class MeanValues(State):
    """
    A :class:`.State` that computes the mean of its values over a given window
    size.
    """

    values_history: torch.Tensor

    def __init__(self, window: int, dtype: torch.dtype | str):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"No a valid data type: {dtype}!")

        self.window = window
        self.register_buffer("values_history", torch.zeros((0,), dtype=dtype), persistent=False)

    def forward(self, values: torch.Tensor) -> None:
        raise NotImplementedError

    def extend(self, values: torch.Tensor) -> None:
        values = values.unsqueeze(0)
        values = values.expand((self.values_history.shape[0], *values.shape))

        self.values_history = torch.cat([self.values_history, values], dim=1)

    def observe(self) -> torch.Tensor:
        return torch.mean(self.values_history, dim=0)

    def reset(self) -> None:
        self.values_history = self.values_history[:0]
