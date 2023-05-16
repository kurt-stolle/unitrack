from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict, TensorDictBase

from .constants import KEY_FRAME, KEY_ID, KEY_INDEX

__all__ = ["Context"]


class Context:
    def __init__(
        self,
        context: Optional[TensorDictBase | Dict[str, Any]],
        detections: Optional[TensorDictBase | Dict[str, Any]],
        frame: int,
        delta: int = 1,
    ) -> None:
        if context is None:
            context = TensorDict.from_dict({}, batch_size=(0,))
        elif isinstance(context, dict):
            context = TensorDict.from_dict(context)
            if len(context.batch_size) == 0:
                context.batch_size = (0,)

        if detections is None:
            detections = TensorDict.from_dict({}, batch_size=(0,))
        elif isinstance(detections, dict):
            detections = TensorDict.from_dict(detections)
            if len(detections.batch_size) == 0:
                detections.batch_size = (0,)

        self.context = context
        self.detections = detections
        self.ids = torch.zeros(detections.batch_size[0], dtype=torch.long, device=detections.device)
        self.frame = frame
        self.delta = delta

    def match(self, cs: TensorDictBase, ds: TensorDictBase) -> None:
        """
        Match candidates to detections. Propagates data and IDs from detections to candidates.

        Parameters
        ----------
        cs
            Candidates
        ds
            Detections
        """
        assert all(cs.get(KEY_FRAME) < self.frame), (cs.get(KEY_FRAME).detach().cpu().tolist(), self.frame)

        for key, value in ds.items():
            if key.startswith("_"):
                continue
            if key not in cs.keys():
                continue
            cs.set_(key, value)
        cs.fill_(KEY_FRAME, self.frame)

        self.ids[ds.get(KEY_INDEX)] = cs.get(KEY_ID)
