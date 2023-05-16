import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tensordict import TensorDict

from unitrack import costs


def test_reduced():
    red = costs.Reduce(
        costs=[
            costs.CategoryGate(),
            costs.CategoryGate(),
        ],
        method=costs.Reduction.SUM,
    )
    res = red(
        TensorDict.from_dict({"categories": torch.arange(3)}), TensorDict.from_dict({"categories": torch.arange(3)})
    )
    assert res.shape == (3, 3), res.shape
    assert res.diag().sum() == 6.0, res.diag().sum()


@settings(deadline=None)
@given(cs_num=st.integers(0, 10), ds_num=st.integers(0, 10))
def test_category_gate(cs_num, ds_num):
    cs = TensorDict.from_dict({"categories": torch.arange(cs_num)})
    assert len(cs) == cs_num, cs

    ds = TensorDict.from_dict({"categories": torch.arange(ds_num)})
    assert len(ds) == ds_num, ds

    gate = costs.CategoryGate()

    gate_matrix: torch.Tensor = gate(cs, ds)
    gate_shape = gate_matrix.shape
    assert gate_matrix.shape[0] == cs_num, gate_shape
    assert gate_matrix.shape[1] == ds_num, gate_shape

    gate_diag = gate_matrix.diag()

    assert gate_diag.sum() == 1.0 * min(ds_num, cs_num), gate_diag

    if cs_num > 1 and ds_num > 1:
        assert gate_matrix.any()
        assert not gate_matrix.all()
    elif cs_num == 1 and ds_num == 1:
        assert gate_matrix.all()


@settings(deadline=None)
@given(cs_num=st.integers(0, 10), ds_num=st.integers(0, 10))
def test_cdist_cost(cs_num: int, ds_num: int):
    x_cs = cs_num * 2.0
    cs = TensorDict.from_dict({"x": torch.arange(cs_num).unsqueeze(1).float() * x_cs})
    assert len(cs) == cs_num, cs
    x_ds = ds_num * 3.0
    ds = TensorDict.from_dict({"x": torch.arange(ds_num).unsqueeze(1).float() * x_ds})
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
    cs = TensorDict.from_dict({"x": cs_x})

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
    ds = TensorDict.from_dict({"x": ds_x})

    cost = costs.MaskIoU(field="x")
    cost_matrix = cost(cs, ds)
    cost_shape = cost_matrix.shape

    assert cost_shape == (len(cs), len(ds)), cost_shape

    print(cost_matrix)
