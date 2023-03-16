from abc import abstractmethod
from typing import Dict, List

import torch

__all__ = ["Field"]


class Field(torch.nn.Module):
    """
    A field is a module that transforms data into state items of a tracked
    object.
    """

    id: str

    def __init__(
        self, id: str, required_keys: List[str], required_data: List[str]
    ):
        """
        Parameters
        ----------
        id
            Unique ID for this field.
        required_data
            List of required keys in the input data.
        """

        super().__init__()

        self.id = id
        self.required_keys = required_keys
        self.required_data = required_data

    def forward(
        self, kvmap: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
    ):
        self.validate_kvmap(kvmap)
        # self.validate_data(data)
        return self.extract(kvmap, data)

    @torch.jit.ignore
    def __repr__(self) -> str:
        req = ", ".join(self.required_keys)
        return f"{type(self).__name__}({self.id}, " f"data=[{req}])"

    def validate_kvmap(self, kvmap: Dict[str, torch.Tensor]) -> None:
        """
        Validate whether all required keys in `data` are present.

        Parameters
        ----------
        data
            Data to validate.

        Raises
        ------
        ValueError
            Missing key in `data`.
        """
        for key in self.required_keys:
            if key in kvmap:
                continue
            raise ValueError(f"Field value with key '{key}' not found!")

    def validate_data(self, data: Dict[str, torch.Tensor]) -> None:
        """
        Validate whether all required keys in `data` are present.

        Parameters
        ----------
        data
            Data to validate.

        Raises
        ------
        ValueError
            Missing key in `data`.
        """
        for key in self.required_data:
            if key in data:
                continue
            raise ValueError(f"Data with key '{key}' not found!")

    @abstractmethod
    def extract(
        self, kvmap: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract field values from data.

        Parameters
        ----------
        data
            Mapping of data keys to tensors.

        Returns
        -------
            Mapping of field names to values.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Update the field state with new values.

        Parameters
        ----------
        state
            Current state of the field.
        values
            New set of values.


        Returns
        -------
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError
