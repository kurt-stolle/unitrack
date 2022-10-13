import torch

from tracking.multitracker import AssociationTracker


def test_association_tracker():
    boxes = [
        [10, 10, 20, 20],
        [30, 30, 40, 40],
        [60, 60, 70, 70],
        [80, 80, 100, 100],
    ]
    embeddings = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=torch.float64,
    )
    scores = [0.9, 0.8, 0.7, 0.2]
    categories = [1, 1, 2, 2]

    tr = AssociationTracker(
        use_kalman=False,
        det_thres=0.6,
        # smooth_embeddings=1.0
    )

    prev_ids: None | list[str] = None

    for frame in range(10):
        tracklets = tr.update(
            frame=frame,
            boxes=torch.tensor(boxes),
            embeddings=embeddings,
            scores=torch.tensor(scores),
            categories=torch.tensor(categories),
        )

        # There are three entries to be tracked, since one has a score below the threhsold
        assert len(tracklets) == 3

        for tl in tracklets:
            assert tl.track_id > 0

        # Since we keep feeding the same data to the tracker, the tracklets
        # should remain the same.
        if prev_ids is None:
            continue

        new_ids = [tr.unique_id for tr in tracklets]

        assert len(new_ids) == len(prev_ids)
        assert all(new == old for new, old in zip(new_ids, prev_ids))

        prev_ids = new_ids
