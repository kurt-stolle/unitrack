from abc import abstractmethod
from typing import Iterable, List

import torch
from tensordict import TensorDict
from tensordict.utils import NestedKey

__all__ = ["Field"]


def _optional_list(xs: Iterable[NestedKey] | None) -> List[NestedKey]:
    if xs is None:
        return []
    return list(x for x in xs)


class Field(torch.nn.Module):
    """
    A field is a module that transforms data into state items of a tracked
    object.
    """

    in_det: torch.jit.Final[List[NestedKey]]
    in_ctx: torch.jit.Final[List[NestedKey]]

    def __init__(
        self,
        *,
        in_det: Iterable[NestedKey] | None = None,
        in_ctx: Iterable[NestedKey] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        in_detection
            List of keys to read from the detection (e.g. object embeddings)
        in_context
            List of keys to read from the context (e.g. image-scale features)
        """

        super().__init__(**kwargs)

        self.in_det = _optional_list(in_det)
        self.in_ctx = _optional_list(in_ctx)

    def __repr__(self) -> str:
        kwvars = ", ".join(
            f"{k}=[{v}]"
            for k, v in {
                "in_detections": ", ".join("_".join(d) for d in self.in_det),
                "in_context": ", ".join("_".join(d) for d in self.in_ctx),
            }.items()
        )

        return f"{type(self).__name__}({self.id}, {kwvars})"
