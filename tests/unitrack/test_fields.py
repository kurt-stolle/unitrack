import pytest
import torch
from tensordict import TensorDict

from unitrack import fields


@pytest.mark.parametrize("dtype", (torch.float32, torch.long, torch.float16, torch.int32, torch.bool))
def test_value_field(dtype):
    data_key = "data_key"

    value_field = fields.Value(key=data_key)

    data_value = torch.arange(9).to(dtype)
    data = {data_key: data_value}

    value_result = value_field(ctx=TensorDict.from_dict({}), det=TensorDict.from_dict(data))

    assert torch.allclose(value_result, data_value)
    assert value_result.dtype == dtype
