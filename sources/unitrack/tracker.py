from typing import Iterable

import numpy as np
import numpy.typing as NP
import torch
from detectron2.structures import Instances

from . import cost, matching
from .constants import START_FRAME
from .kalman import KalmanFilter
from .tracklet import Tracklet, TrackState
from .utils.box import tlbr_to_tlwh

__all__ = ["Tracker"]


class Tracker:
    def __init__(
        self,
        *,
        max_track_id=100000,
        assignment_thres=0.8,
        max_time_lost=5,
        buffer_size=30,
        det_thres=0.5,
        det_beta=0.9,
        use_kalman=True,
        asso_with_motion=True,
        smooth_embeddings=1.0,
    ):
        assert 0.0 < det_beta <= 1.0, f"Invalid detection beta: {det_beta}"

        # Current frame
        self.frame = START_FRAME - 1

        # Tracklets
        self.tracklets: list[Tracklet] = []

        # Hyperparameters
        self.use_kalman = use_kalman
        self.asso_with_motion = asso_with_motion
        self.det_thres = det_thres
        self.det_beta = det_beta
        self.buffer_size = buffer_size
        self.max_time_lost = max_time_lost
        self.assignment_thres = assignment_thres
        self.max_track_id = max_track_id

        self.track_id_pool = {}
        self.smooth_embeddings = smooth_embeddings

        self.kalman_filter = KalmanFilter()

        self.motion_lambda = 0.98
        self.motion_gated = False

    # def get_candidate_tracks(self) -> tuple[list[Tracklet], list[Tracklet]]:
    #     """
    #     Get a list of candidate tracks and unconfirmed candidate tracks
    #     """
    #     tracks_unconfirmed = []
    #     tracks_active = []
    #     for t in self.tracked:
    #         if not t.is_activated:
    #             tracks_unconfirmed.append(t)
    #         else:
    #             tracks_active.append(t)

    #     tracks = joint_stracks(tracks_active, self.lost)

    #     return tracks, tracks_unconfirmed

    def update(
        self,
        frame: int,
        boxes: NP.ArrayLike,
        embeddings: torch.Tensor,
        scores: NP.ArrayLike,
        categories: NP.ArrayLike,
    ) -> list[Tracklet]:
        # torch.cuda.empty_cache()

        # Save the current frame number
        assert self.frame < frame, (
            f"Attempted to track back in time from frame  {self.frame} "
            f"to {frame}."
        )
        self.frame = frame

        # Detections list
        detections = (
            Tracklet(
                tlwh=tlbr_to_tlwh(box),
                score=score,
                # temp_feat=emb.reshape(8, 256 // 8),
                temp_feat=emb,
                category=cat,
                index=index,
                smooth=self.smooth_embeddings,
                use_kalman=self.use_kalman,
            )
            for index, (box, score, emb, cat) in enumerate(
                zip(
                    np.asarray(boxes),
                    np.asarray(scores),
                    embeddings,
                    np.asarray(categories),
                )
            )
        )
        detections = list(
            d for d in detections if d.score > self.det_thres * self.det_beta
        )

        # Candidates are all tracklets in memory that are either lost or being
        # tracked
        candidates = [
            t
            for t in self.tracklets
            if t.state in (TrackState.TRACKING, TrackState.LOST)
        ]

        # Compute assignment cost matrix
        # cost_matrix, recons_ftrk = matching.reconsdot_distance(
        #     tracks, detections
        # )
        cost_matrix = cost.embedding_distance(candidates, detections)

        # When using the Kalman filter, add motion costs
        if self.use_kalman:
            # Predict the current location with KF
            Tracklet.multi_predict(self.kalman_filter, candidates)

            # Add cost based on the distance between the predicted location
            # and each detection's current location
            cost_matrix = matching.fuse_motion(
                self.kalman_filter,
                cost_matrix,
                candidates,
                detections,
                lambda_=self.motion_lambda,
                gate=self.motion_gated,
            )

        # Peform first assocication
        candidates, detections = self.associate(
            candidates, detections, cost_matrix, self.assignment_thres
        )

        # Match remaining tracks using IoU
        if self.asso_with_motion:
            # Drop all candidates that are not currently tracked
            candidates = [
                c for c in candidates if c.state == TrackState.TRACKING
            ]

            # Compute distance matrix using IoU between tracks and detections
            cost_matrix = cost.iou_distance(candidates, detections)

            # Perform second assocication
            self.associate(candidates, detections, cost_matrix, threshold=0.5)

        # Mark all tracks from memory that have not been matched to a track
        # in the current frame as lost.
        for c in candidates:
            c.state = TrackState.LOST

        # For all remaining detections, which are thus not associated with a
        # candidate track from previous frames, a new track is stated
        for d in detections:
            # If the `score` entry of the track, which can be interpreted as
            # the confidence of detection, is above a threshold value, then
            # activate the track.
            if d.score < self.det_thres:
                continue

            track_id = self.track_id_pool.setdefault(
                d.category, list(range(1, self.max_track_id))
            ).pop(0)

            d.activate(
                kalman_filter=self.kalman_filter,
                frame=self.frame,
                track_id=track_id,
            )

            self.tracklets.append(d)

        # Remove tracks that have been lost for a configured amount of frames
        self.remove_lost()

        # Remove duplicates
        # self.tracked, self.lost = remove_duplicate_stracks(
        #     self.tracked,
        #     self.lost,
        #     ioudist=self.dup_iou_thres,
        # )

        # Output only activated tracks
        return [t for t in self.tracklets if t.state == TrackState.TRACKING]

    def update_with_instances(self, frame: int, ins: Instances):
        """
        Update using a Detectron2 Instances structure
        """

        for f in ("pred_boxes", "embeddings", "pred_classes", "scores"):
            if f in ins.get_fields():
                continue
            raise ValueError(f"Missing field in Instances object: {f}")

        return self.update(
            frame=frame,
            boxes=ins.pred_boxes.tensor.numpy(),
            embeddings=ins.embeddings,
            categories=ins.pred_classes.numpy(),
            scores=ins.scores.numpy(),
        )

    def associate(
        self,
        candidates: list[Tracklet],
        detections: list[Tracklet],
        cost_matrix: NP.NDArray[np.float64],
        threshold: float,
    ):
        assert cost_matrix.shape == (len(candidates), len(detections))

        # Mask the distance matrix entries of tracks with different categories
        cost_matrix = matching.category_gate(
            cost_matrix, candidates, detections
        )

        # Solve linear assignment problem
        matches, ic_unmatch, id_unmatch = matching.linear_assignment(
            cost_matrix, thresh=threshold
        )

        # For each match, assign the track/candidate pair
        for ic, id in matches:
            c = candidates[ic]
            d = detections[id]
            if c.state == TrackState.TRACKING:
                c.update(d, self.frame)
            elif c.state == TrackState.LOST:
                c.re_activate(d, self.frame)
            else:
                raise ValueError(f"Cannot match tracklet with state {c.state}")

        # Return new list of unmatched candidates/detections
        return list(candidates[i] for i in ic_unmatch), list(
            detections[i] for i in id_unmatch
        )

    def remove_lost(self):
        for c in self.tracklets:
            if c.state != TrackState.LOST:
                continue
            if self.frame - c.end_frame > self.max_time_lost:
                c.state = TrackState.REMOVED
