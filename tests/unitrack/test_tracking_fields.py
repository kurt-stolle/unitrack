from typing import cast

import pytest
import torch
from unitrack import fields


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dtype", (torch.float32, torch.long, torch.float32))
def test_value_field(dtype, jit):
    data_key = "data_key"

    value_field = fields.Value(id=data_key)
    if jit:
        value_field = cast(fields.Value, torch.jit.script(value_field))

    data_value = torch.arange(9).to(dtype)
    data = {data_key: data_value}

    value_result = value_field(data, {})

    assert torch.allclose(value_result, data_value)
    assert value_result.dtype == dtype


@pytest.mark.parametrize("jit", [False, True])
def test_projection_field(jit):
    id = "id"
    key_mask = "mask"
    key_depth = "depth"

    value_field = fields.DepthProjection(
        id=id, key_mask=key_mask, key_depth=key_depth, max_depth=100.0
    )
    if jit:
        value_field = cast(
            fields.DepthProjection, torch.jit.script(value_field)
        )

    data = {
        key_mask: torch.arange(3 * 3).reshape(3, 3).bool()[None, :],
        key_depth: torch.arange(3).float()[None, :],
    }

    assert data[key_mask].ndim == 3
    assert data[key_depth].ndim == 2

    value_result = value_field(data, {})

    assert value_result is not None
