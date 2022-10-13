import pickle
from pathlib import Path

import pytest

from tracking import Tracklet

DATA_PATH = Path("./assets/test-data/tracking").resolve()


def read_pickle(p: Path):
    with open(p, "rb") as fh:
        data = pickle.load(fh)
    assert data is not None
    return data


def pred_things_data():
    boxes = []
    classes = []
    embeddings = []
    masks = []
    scores = []

    for item in DATA_PATH.iterdir():
        if not item.is_dir():
            continue
        boxes.append(read_pickle(item / "pred_boxes.pkl"))
        classes.append(read_pickle(item / "pred_classes.pkl"))
        embeddings.append(read_pickle(item / "pred_embeddings.pkl"))
        masks.append(read_pickle(item / "pred_masks.pkl"))
        scores.append(read_pickle(item / "pred_scores.pkl"))

    return boxes, classes, embeddings, masks, scores


@pytest.fixture(scope="session")
def detections():
    dets_all = []
    for pred_things in zip(*pred_things_data()):
        dets_frame = []
        for index, (box, cat, embedding, _, score) in enumerate(
            zip(*pred_things)
        ):
            dets_frame.append(
                Tracklet(
                    tlwh=box,
                    category=cat,
                    score=score,
                    temp_feat=embedding,
                    index=index,
                )
            )

        dets_all.append(dets_frame)

    return dets_all


@pytest.fixture(scope="session")
def pred_sem_segs():
    return [
        read_pickle(item / "pred_sem_seg.pkl")
        for item in DATA_PATH.iterdir()
        if item.is_dir()
    ]


@pytest.fixture(scope="session")
def true_labels():
    return [
        read_pickle(item / "true_labels.pkl")
        for item in DATA_PATH.iterdir()
        if item.is_dir()
    ]
