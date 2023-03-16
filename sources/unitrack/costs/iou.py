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

from ..detections import Detections
from .base_cost import Cost


@torch.jit.script
def _pad_degenerate_boxes(boxes: Tensor):
    """
    Adds 1 to the far-coordinate of each box to prevent degeneracy from mask to
    box conversion.
    """
    assert boxes.ndim == 2

    boxes = boxes.clone()
    boxes[:, 2] += 1
    boxes[:, 3] += 1

    return boxes


@torch.jit.script
def _upcast(t: Tensor) -> Tensor:
    """
    Protects from numerical overflows in multiplications by upcasting to the
    equivalent higher type.
    """

    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


@torch.jit.script
def _box_diou_iou(
    boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7
) -> Tuple[Tensor, Tensor]:
    """
    IoU with penalized center-distance
    """

    iou = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    # centers of boxes
    x_p = (boxes1[:, 0] + boxes1[:, 2]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 3]) / 2
    x_g = (boxes2[:, 0] + boxes2[:, 2]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # The distance between boxes' centers squared.
    centers_distance_squared = (_upcast((x_p[:, None] - x_g[None, :])) ** 2) + (
        _upcast((y_p[:, None] - y_g[None, :])) ** 2
    )

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared), iou


@torch.jit.script_if_tracing
def _complete_box_iou(
    boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7
) -> Tensor:
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)

    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    w_pred = boxes1[:, None, 2] - boxes1[:, None, 0]
    h_pred = boxes1[:, None, 3] - boxes1[:, None, 1]

    w_gt = boxes2[:, 2] - boxes2[:, 0]
    h_gt = boxes2[:, 3] - boxes2[:, 1]

    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return diou - alpha * v


class MaskIoU(Cost):
    """
    Computes IoU cost matrix between two sets of bitmasks.
    """

    def __init__(self, field: str):
        super().__init__(required_fields=(field,))

        self.field_name: Final = field

    def compute(self, cs: Detections, ds: Detections) -> Tensor:
        cs_m = cs.get(self.field_name).reshape(len(cs), -1).int()
        ds_m = ds.get(self.field_name).reshape(len(ds), -1).int()

        isec = torch.mm(cs_m, ds_m.T)
        area = cs_m.sum(dim=1)[:, None] + ds_m.sum(dim=1)[None, :]

        eps = 1e-6
        iou = (isec + eps) / (area - isec + eps)

        return 1.0 - iou

        # cost_matrix = torch.full(
        #     (len(cs_field), len(ds_field)), torch.nan, device=cs_field.device
        # )

        # for ds_idx, ds_box in enumerate(ds_field):
        #     for cs_idx, cs_box in enumerate(cs_field):
        #         cost_matrix[cs_idx, ds_idx] = generalized_box_iou_loss(
        #             ds_box, cs_box
        #         )

        # return cost_matrix


class BoxIoU(Cost):
    """
    Computes the generalized IoU cost matrix between two sets of bounding boxes.
    """

    def __init__(self, field: str):
        super().__init__(required_fields=(field,))

        self.field: Final = field

    def compute(self, cs: Detections, ds: Detections) -> Tensor:
        cs_field = _pad_degenerate_boxes(cs.get(self.field))
        ds_field = _pad_degenerate_boxes(ds.get(self.field))

        return 1.0 - _complete_box_iou(cs_field, ds_field)
