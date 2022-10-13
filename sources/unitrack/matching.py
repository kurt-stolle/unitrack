import lap
import numpy as np
import numpy.typing as NP
from scipy.sparse import coo_matrix

from .kalman import KalmanFilter
from .tracklet import Tracklet


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(
    cost_matrix, thresh
) -> (np.ndarray, np.ndarray, np.ndarray):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def fuse_motion(
    kf: KalmanFilter,
    cost_matrix: NP.NDArray[np.float64],
    tracks: list[Tracklet],
    detections: list[Tracklet],
    only_position=False,
    lambda_=0.98,
    gate=True,
):
    """
    Augment cost matrix with motion costs
    """
    if cost_matrix.size == 0:
        return cost_matrix

    # First two dimensions are position
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]

    # Measurements are the bounding boxes of the detections
    measurements = np.asarray([det.to_xyah() for det in detections])

    # For each track, compute the gating distance betweeen it and each
    # measurement
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
            metric="maha",
        )

        # If the gating distance exceeds the threshold, set cost to infinite
        if gate:
            cost_matrix[row, gating_distance > gating_threshold] = np.inf

        # Apply cost based on distance
        cost_matrix[row] = (
            lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
        )

    # Return augmented cost matrix
    return cost_matrix


def category_gate(cost_matrix, tracks, detections):
    """
    Category gate

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix
    tracks : list[Tracklet]
        Tracks
    detections : list[Tracklet]
        Detections

    Returns
    -------
    np.ndarray
        Gated cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix

    det_categories = np.array([d.category for d in detections])
    trk_categories = np.array([t.category for t in tracks])

    # Maximize cost for matrix entires with unequal category
    # cost_matrix = cost_matrix + np.abs(
    #     det_categories[None, :] - trk_categories[:, None]
    # )

    cost_matrix = np.where(
        (det_categories[None, :] - trk_categories[:, None]) == 0,  # type: ignore
        cost_matrix,
        np.inf,
    )

    return cost_matrix
