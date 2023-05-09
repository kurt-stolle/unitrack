import torch
from unitrack import Detections


def test_detections():
    field_id = "field_id"
    value = torch.arange(6)

    ds = Detections({"field_id": value})

    assert len(ds) == len(value)

    for slice in range(len(value)):
        ds_b = ds[:slice]
        ds_t = ds[slice:]

        assert isinstance(ds_b, Detections), type(ds_split)
        assert isinstance(ds_t, Detections), type(ds_split)

        value_b = value[:slice]
        value_t = value[slice:]

        assert len(value_b) == len(ds_b)
        assert len(value_t) == len(ds_t)

        assert torch.allclose(value_b, ds_b.get(field_id))
        assert torch.allclose(value_t, ds_t.get(field_id))
