from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase

__all__ = ["Frame"]

from typing import Dict, TypeAlias

MaybeTensorDict: TypeAlias = TensorDictBase | Dict[str, torch.Tensor]


class Frame:
    def __init__(
        self,
        detections: MaybeTensorDict,
        *,
        context: MaybeTensorDict | None = None,
        frame: int = 0,
        delta: int = 1,
        key: str | None = None,
    ) -> None:
        self.detections = self._make_tensordict(detections)
        assert self.detections.device is not None, "Device of detections is None!"

        self.context = self._make_tensordict(context)
        self.ids = torch.zeros(self.detections.batch_size[0], dtype=torch.long, device=self.detections.device)
        self.key = key
        self.frame = frame
        self.delta = delta

    @staticmethod
    def _make_tensordict(d: MaybeTensorDict | None) -> TensorDictBase:
        if d is None:
            return TensorDict.from_dict({}, batch_size=(0,))
        if isinstance(d, TensorDictBase):
            return d
        return TensorDict.from_dict(d)
