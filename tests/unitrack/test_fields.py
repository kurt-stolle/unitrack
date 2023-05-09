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
