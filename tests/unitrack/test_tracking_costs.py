
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from unitrack import Detections, costs


@settings(deadline=None)
@given(cs_num=st.integers(0, 10), ds_num=st.integers(0, 10))
def test_category_gate(cs_num, ds_num):
    cs = Detections({"categories": torch.arange(cs_num)})
    assert len(cs) == cs_num, cs

    ds = Detections({"categories": torch.arange(ds_num)})
    assert len(ds) == ds_num, ds

    gate = costs.CategoryGate()

    gate_matrix: torch.Tensor = gate(cs, ds)
    gate_shape = gate_matrix.shape
    assert gate_matrix.shape[0] == cs_num, gate_shape
    assert gate_matrix.shape[1] == ds_num, gate_shape

    gate_diag = gate_matrix.diag()

    assert gate_diag.sum() == 1.0 * min(ds_num, cs_num), gate_diag

    if cs_num > 1 and ds_num > 1:
        gate_max = gate_matrix.max()
        gate_min = gate_matrix.min()
        assert gate_max.max() == torch.inf, gate_max
        assert gate_min == 1.0, gate_min
    elif cs_num == 1 and ds_num == 1:
        gate_max = gate_matrix.max()
        gate_min = gate_matrix.min()
        assert gate_min == gate_max == 1.0, (gate_min, gate_max)


@settings(deadline=None)
@given(cs_num=st.integers(0, 10), ds_num=st.integers(0, 10))
def test_cdist_cost(cs_num: int, ds_num: int):
    x_cs = cs_num * 2.0
    cs = Detections({"x": torch.arange(cs_num).unsqueeze(1).float() * x_cs})
    assert len(cs) == cs_num, cs
    x_ds = ds_num * 3.0
    ds = Detections({"x": torch.arange(ds_num).unsqueeze(1).float() * x_ds})
    assert len(ds) == ds_num, ds

    cost = costs.Distance(field="x")

    cost_matrix = cost(cs, ds)
    cost_shape = cost_matrix.shape
    assert cost_shape == (len(cs), len(ds)), cost_shape

    for cs_i in range(cs_num):
        for ds_i in range(ds_num):
            euclidean_distance = abs((cs_i * x_cs) - (ds_i * x_ds))
            assert cost_matrix[cs_i, ds_i] == euclidean_distance


def test_mask_iou():
    cs_x = torch.tensor(
        [
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ],
        dtype=torch.int,
    )
    cs = Detections({"x": cs_x})

    ds_x = torch.tensor(
        [
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ],
        dtype=torch.int,
    )
    ds = Detections({"x": ds_x})

    cost = costs.MaskIoU(field="x")
    cost_matrix = cost(cs, ds)
    cost_shape = cost_matrix.shape

    assert cost_shape == (len(cs), len(ds)), cost_shape

    print(cost_matrix)

