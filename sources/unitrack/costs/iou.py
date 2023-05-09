"""
IoU-related assignment cost matrices

Some utility methods in this package are copied from the ``torchvision`` 
package, as these utility methods have not yet been merged to the release
version of that package.
"""


from typing import Final, Tuple

import torch
from torch import Tensor
from torchvision.ops import box_iou

from ..structures import Detections
from .base_cost import Cost

__all__ = ["MaskIoU", "BoxIoU"]

DEFAULT_EPS: Final = 1e-8


class MaskIoU(Cost):
    """
    Computes IoU cost matrix between two sets of bitmasks.
    """

    eps: torch.jit.Final[float]
    field: torch.jit.Final[str]

    def __init__(self, field: str, eps: float = DEFAULT_EPS):
        super().__init__(required_fields=(field,))

        self.field: Final = field
        self.eps = eps

    def compute(self, cs: Detections, ds: Detections) -> Tensor:
        return _naive_mask_iou(cs.get(self.field), ds.get(self.field), self.eps)


class BoxIoU(Cost):
    """
    Computes the generalized IoU cost matrix between two sets of bounding boxes.
    """

    field: torch.jit.Final[str]
    eps: torch.jit.Final[float]

    def __init__(self, field: str, eps: float = DEFAULT_EPS):
        super().__init__(required_fields=(field,))

        self.field = field
        self.eps = eps

    def compute(self, cs: Detections, ds: Detections) -> Tensor:
        cs_field = _pad_degenerate_boxes(cs.get(self.field))
        ds_field = _pad_degenerate_boxes(ds.get(self.field))

        return 1.0 - _complete_box_iou(cs_field, ds_field, self.eps)


def _naive_mask_iou(cs: Tensor, ds: Tensor, eps: float) -> Tensor:
    cs_m = cs.reshape(len(cs), -1).int()
    ds_m = ds.reshape(len(ds), -1).int()

    isec = torch.mm(cs_m, ds_m.T)
    area = cs_m.sum(dim=1)[:, None] + ds_m.sum(dim=1)[None, :]

    iou = (isec + eps) / (area - isec + eps)

    return 1.0 - iou


def _pad_degenerate_boxes(boxes: Tensor) -> Tensor:
    """
    Adds 1 to the far-coordinate of each box to prevent degeneracy from mask to
    box conversion.
    """
    boxes = boxes.clone()
    boxes[:, 2] += 1
    boxes[:, 3] += 1

    return boxes


def _box_diou_iou(boxes1: Tensor, boxes2: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
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


def _complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps) -> Tensor:
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
