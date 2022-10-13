from pathlib import Path

import numpy as np
from fixture_data import detections
from unitrack import Tracklet, cost


def test_iou_distance(detections):
    for a, b in zip(detections[1:], detections[:-1]):
        dist = cost.iou_distance(a, b)
        assert dist.shape == (len(a), len(b))


def test_emb_distance(detections):
    for a, b in zip(detections[1:], detections[:-1]):
        dist = cost.embedding_distance(a, b)
        assert dist.shape == (len(a), len(b))
