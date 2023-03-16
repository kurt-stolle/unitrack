from typing import Any, Dict, Optional

import torch


class Detections:
    """
    Detections is a container class for field and state values for detections at
    current (field values) and previous (state values) timesteps.
    """

    def __init__(
        self,
        fields: Dict[str, torch.Tensor],
        indices: Optional[torch.Tensor] = None,
        mutable: Optional[bool] = None,
    ):
        self._fields = fields

        if mutable is None:
            mutable = False
        self.mutable = mutable

        if indices is None:
            device: Optional[torch.device] = None
            for value in fields.values():
                device = value.device
                break
            if device is None:
                raise ValueError("Failed to find detections device!")
            indices = torch.arange(len(self), device=device, dtype=torch.long)
        self.indices = indices

        self._validate()

    def __getitem__(self, index):
        select: Dict[str, torch.Tensor] = {}
        for key, values in self._fields.items():
            if isinstance(values, torch.Tensor):
                select[key] = values[index]
            else:
                raise TypeError(f"Unsupported field type at key '{key}'!")
        return Detections(
            fields=select, indices=self.indices[index], mutable=False
        )

    def __len__(self):
        for item in self._fields.values():
            return len(item)
        raise NotImplementedError(f"Empty detections class!")

    def __contains__(self, key: str) -> bool:
        return key in self._fields.keys()

    def get(
        self,
        key: str,
    ) -> torch.Tensor:
        return self._fields[key]

    def set(self, key: str, value: torch.Tensor) -> None:
        if not self.mutable:
            raise RuntimeError("Not a mutable collection!")
        if key not in self:
            raise ValueError(f"Key '{key}' not in {self}")

        self._fields[key] = value

    def __getattr__(self, key: str) -> Any:
        if key == "indices":
            return self.indices
        if key == "mutable":
            return self.mutable
        return self._fields[key]

    def _validate(self) -> None:
        len_self = len(self)
        if len(self.indices) != len(self):
            raise ValueError(
                f"Indices length {len(self.indices)} not equal to fields "
                f"length {len_self}!"
            )

        for key, value in self._fields.items():
            len_value = len(value)
            if len_self != len_value:
                raise ValueError(
                    f"Excepted fields of length {len_self} but found '{key}' "
                    f"with length {len_value}!"
                )

    def __repr__(self) -> str:
        indices: list[int] = self.indices.tolist()
        field_names: list[str] = list(self._fields.keys())
        return (
            f"Detections({len(self)}, "
            f"fields={field_names}, "
            f"indices={indices}, mutable={self.mutable})"
        )
