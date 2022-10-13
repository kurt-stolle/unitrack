import numpy as np

from tracking.association import matching
from tracking.tracklet import Tracklet


def test_category_gate():

    tlwh = [1, 1, 2, 2]
    tracklets = [
        Tracklet(tlwh=tlwh, score=1.0, temp_feat=[], index=0, category=1),
        Tracklet(tlwh=tlwh, score=1.0, temp_feat=[], index=1, category=1),
        Tracklet(tlwh=tlwh, score=1.0, temp_feat=[], index=2, category=3),
    ]

    cost = np.ones((3, 3))
    cost_gated = matching.category_gate(cost, tracklets, tracklets)

    cost_true = np.array(
        [[1.0, 1.0, np.inf], [1.0, 1.0, np.inf], [np.inf, np.inf, 1.0]]
    )

    assert (cost_gated == cost_true).all()


def test_assignment():
    cost_matrix = np.arange(3 * 3).reshape(3, 3)
    cost_matrix[2, :] *= 100
    cost_matrix[1, 1] *= 100

    match, unmatch_a, unmatch_b = matching.linear_assignment(cost_matrix, 10.0)

    assert (match == np.array([[0, 1], [1, 0]])).all()
    assert (unmatch_a == unmatch_b).all()
