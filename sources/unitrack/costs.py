"""
This package implements the cost functions that are used to compute the assignment costs between tracklets and
detections.
"""

from abc import abstractmethod

import typing as T
import enum as E
import itertools
import torch
import torch.nn as nn
from tensordict import TensorDictBase


__all__ = []


class Cost(torch.nn.Module):
    """
    A cost module computes an assignment cost matrix between detections and
    tracklets.
    """

    required_fields: torch.jit.Final[list[str]]

    def __init__(self, required_fields: T.Iterable[str]):
        super().__init__()

        self.required_fields = list(set(required_fields))

    @abstractmethod
    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        """
        Computes the assignment costs between previous tracklets and current
        detections.

        This is an abstract method that should be overwritten.

        Parameters
        ----------
        cs
            Candidate observations (N).
        ds
            Detections (M).
        Returns
        -------
            Cost matrix (N x M).
        """
        raise NotImplementedError


class FieldCost(Cost):
    field: torch.jit.Final[str]

    def __init__(self, field: str):
        super().__init__(required_fields=[field])

        self.field = field

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        return self.compute(cs.get(self.field), ds.get(self.field))

    @abstractmethod
    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        """
        Computes the assignment costs between previous tracklets and current
        detections.

        This is an abstract method that should be overwritten.

        Parameters
        ----------
        cs
            Candidate observations (N) values for a single field.
        ds
            Detections (M) values for a single field.

        Returns
        -------
            Cost matrix (N x M).
        """
        raise NotImplementedError


class CategoryGate(FieldCost):
    """
    Returns a matrix where each tracklet/detection pair that have equal
    categories are set to `True` and  that have different categories are set to
    `False`.
    """

    def __init__(self, field="categories"):
        super().__init__(field=field)

    def compute(self, cs_cats: torch.Tensor, ds_cats: torch.Tensor) -> torch.Tensor:
        gate_matrix = ds_cats[None, :] - cs_cats[:, None]

        return gate_matrix == 0


DEFAULT_EPS: T.Final = 1e-8


class MaskIoU(FieldCost):
    """
    Computes IoU cost matrix between two sets of bitmasks.
    """

    eps: torch.jit.Final[float]

    def __init__(self, eps: float = DEFAULT_EPS, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        return _naive_mask_iou(cs, ds, self.eps)


class BoxIoU(MaskIoU):
    def compute(self, cs: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        cs_field = _pad_degenerate_boxes(cs)
        ds_field = _pad_degenerate_boxes(ds)

        return 1.0 - _complete_box_iou(cs_field, ds_field, self.eps)


def _naive_mask_iou(cs: torch.Tensor, ds: torch.Tensor, eps: float) -> torch.Tensor:
    cs_m = cs.reshape(len(cs), -1).int()
    ds_m = ds.reshape(len(ds), -1).int()

    isec = torch.mm(cs_m, ds_m.T)
    area = cs_m.sum(dim=1)[:, None] + ds_m.sum(dim=1)[None, :]

    iou = (isec + eps) / (area - isec + eps)

    return 1.0 - iou


def _pad_degenerate_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """
    Adds 1 to the far-coordinate of each box to prevent degeneracy from mask to
    box conversion.
    """
    boxes = boxes.clone()
    boxes[:, 2] += 1
    boxes[:, 3] += 1

    return boxes


def _box_diou_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    IoU with penalized center-distance
    """

    iou = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = (rbi - lti).float().clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2)

    # centers of boxes
    x_p = (boxes1[:, 0] + boxes1[:, 2]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 3]) / 2
    x_g = (boxes2[:, 0] + boxes2[:, 2]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # The distance between boxes' centers squared.
    centers_distance_squared = (((x_p[:, None] - x_g[None, :])).float() ** 2) + (
        ((y_p[:, None] - y_g[None, :])).float() ** 2
    )

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - ((centers_distance_squared) / diagonal_distance_squared.clamp(eps)), iou


def _complete_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps) -> torch.Tensor:
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    w_pred = boxes1[:, None, 2] - boxes1[:, None, 0]
    h_pred = boxes1[:, None, 3] - boxes1[:, None, 1]

    w_gt = boxes2[:, 2] - boxes2[:, 0]
    h_gt = boxes2[:, 3] - boxes2[:, 1]

    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2)

    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(eps)

    return diou - alpha * v


class Weighted(Cost):
    """
    A weighted cost module.
    """

    def __init__(self, cost: Cost, weight: float):
        super().__init__(required_fields=cost.required_fields)

        self.cost = cost
        self.weight = weight

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        return self.weight * self.cost(cs, ds)


class Reduction(E.Enum):
    """
    Reduction method for :class:`.Reduce` cost module.
    """

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"


class Reduce(Cost):
    """
    A cost reduction module.
    """

    weights: torch.Tensor
    method: torch.jit.Final[Reduction]

    def __init__(
        self,
        costs: T.Sequence[Cost],
        method: Reduction,
        weights: T.Optional[T.Sequence[float]] = None,
    ):
        super().__init__(required_fields=itertools.chain(*(c.required_fields for c in costs)))

        if isinstance(method, str):
            method = Reduction(method)

        self.method = method
        self.costs = nn.ModuleList(costs)
        if weights is None:
            weights = [1.0] * len(costs)

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).unsqueeze_(-1).unsqueeze_(-1))

    def forward(self, cs: TensorDictBase, ds: TensorDictBase) -> torch.Tensor:
        costs = torch.stack([cost(cs, ds) for cost in self.costs])
        costs = costs * self.weights
        costs = _reduce_stack(costs, self.method)

        return costs


@torch.jit.script_if_tracing
def _reduce_stack(costs: torch.Tensor, method: Reduction) -> torch.Tensor:
    if method == Reduction.SUM:
        return costs.sum(dim=0)
    if method == Reduction.MEAN:
        return costs.mean(dim=0)
    if method == Reduction.MIN:
        return costs.min(dim=0).values
    if method == Reduction.MAX:
        return costs.max(dim=0).values
    raise NotImplementedError(f"Reduction method '{method}' not implemented!")


class Distance(Cost):
    """
    Cost function that wraps a distance module.
    """

    select: torch.jit.Final[T.List[int]]
    p: torch.jit.Final[float]
    field: torch.jit.Final[str]

    def __init__(
        self,
        field: str,
        select: T.Optional[T.List[int]] = None,
        p_norm: float = 2.0,
    ):
        """
        Parameters
        ----------
        critereon
            Module that computes distance between two fields
        field_name
            Name if the field to read tensors from that are passed to the
            ``torch.cdist`` function.
        select
            List of indices to select from the field.
        p_norm
            Value of p-norm.
        """
        super().__init__(required_fields=(field,))

        self.field = field

        if select is None:
            select = []

        self.select = select  # type: ignore
        self.p = p_norm

    def get_field(self, cs: TensorDictBase, ds: TensorDictBase) -> T.Tuple[torch.Tensor, torch.Tensor]:
        ts_field = cs.get(self.field)
        ds_field = ds.get(self.field)

        if len(self.select) > 0:
            assert ts_field.shape[1] >= max(self.select)
            assert ds_field.shape[1] >= max(self.select)

            ts_field = ts_field[:, self.select]
            ds_field = ds_field[:, self.select]

        return ts_field, ds_field

    def forward(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return torch.cdist(ts_field, ds_field, self.p, "donot_use_mm_for_euclid_dist")


class Cosine(Distance):
    r"""
    Computes the distance between two fields based on a similarity measure with
    range $[0,1]$ by computing $1 - \mathrm{CosineSimilarity}(x,y)$.
    """

    def __init__(self, *args, eps: float = DEFAULT_EPS, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def compute(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return 1.0 - _cosine_similarity(ts_field, ds_field, self.eps)


class Softmax(Distance):
    r"""
    Computes the distance between two fields based on a similarity measure with
    range $[0,1]$ by computing $1 - \mathrm{ReciprocalSoftmax}(x,y)$.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, cs: TensorDictBase, ds: TensorDictBase):
        ts_field, ds_field = self.get_field(cs, ds)
        return 1.0 - _softmax_similarity(ts_field, ds_field)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps) -> torch.Tensor:
    a_norm = _stable_norm(a, eps)
    b_norm = _stable_norm(b, eps)

    return a_norm @ b_norm.mT


def _stable_norm(t: torch.Tensor, eps):
    norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
    return t / norm.clamp_min(eps)


def _softmax_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mul = torch.mm(a, b.T)
    a2b = mul.softmax(dim=0)
    b2a = mul.softmax(dim=1)
    return (a2b + b2a) / 2.0
