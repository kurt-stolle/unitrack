from __future__ import annotations

import typing as T
from abc import abstractmethod
from typing import Iterable, List, Tuple

import torch
from tensordict import TensorDictBase

from ..constants import DEBUG, KEY_ID, KEY_INDEX

__all__ = ["Stage"]


class Stage(torch.nn.Module):
    """
    Base class for stages in a ::class::`..Tracker`.

    Inputs to a stage are in the context data, or as fields of the detections.
    """

    def __init__(
        self,
        required_fields: Iterable[str] = (),
    ):
        super().__init__()

        self.required_fields = list(required_fields)

    def __repr__(self) -> str:
        req = ", ".join(self.required_fields)
        return f"{type(self).__name__}(fields=[{req}])"

    @abstractmethod
    def forward(
        self,
        ctx: TensorDictBase,
        cs: TensorDictBase,
        ds: TensorDictBase,
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        raise NotImplementedError

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
        if DEBUG:
            print(
                f" - matched {cs.batch_size[0]} candidates to "
                f"{ds.batch_size[0]} detections"
            )

        cs_keys = T.cast(T.List[str], cs.keys())
        ds_keys = T.cast(T.List[str], ds.keys())
        for key in ds_keys:
            if key.startswith("_"):
                continue
            if key not in cs_keys:
                continue
            cs.set_(key, ds.get(key))

        # pass the index of the detection such that it can be retrieved later when
        # collecting the retuned ID
        indices = ds.get(KEY_INDEX)
        cs.set_(KEY_INDEX, ds.get(KEY_INDEX))
        ds.set_(KEY_INDEX, torch.full_like(indices, -1))
