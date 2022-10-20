from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np
import numpy.typing as NP

from . import cost
from .kalman import KalmanFilter
from .utils.box import tlwh_to_xyah

__all__ = ["TrackState", "Tracklet"]


class TrackState(Enum):
    NEW = 1
    TRACKING = 2
    LOST = 3
    REMOVED = 4


class Tracklet:
    def __init__(
        self,
        tlwh: NP.ArrayLike,
        score: float,
        temp_feat,
        index: int,
        buffer_size=30,
        category=-1,
        use_kalman=True,
        smooth=0.9,
    ):
        self.state = TrackState.NEW
        self.track_id = 0
        self.start_frame = -1
        self.frame_id = -1
        self.index = index
        self.score = score
        self.category = category
        self.tracklet_len = 0

        # Bounding box
        self._tlwh = np.asarray(tlwh, dtype=np.float64)

        # Kalman filter
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.use_kalman = use_kalman

        # Features and smoothing
        self.smooth = smooth
        self.smooth_feat = None
        self.curr_feature = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)

    @property
    def unique_id(self):
        return f"{self.category}-{self.track_id}"

    @property
    def end_frame(self):
        return self.frame_id

    def update_features(self, feat):
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        elif self.smooth_feat.shape == feat.shape:
            self.smooth_feat = (
                self.smooth * self.smooth_feat + (1 - self.smooth) * feat
            )
        else:
            # pass
            raise ValueError(
                f"Shape of smoothed features {self.smooth_feat.shape} is not "
                f"equal to new features {feat.shape}."
            )

    def predict(self):
        assert self.kalman_filter is not None
        assert self.mean is not None

        mean_state = self.mean.copy()
        if self.state != TrackState.TRACKING:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(kalman_filter: KalmanFilter, tracklets: list[Tracklet]):
        """
        Predict multiple trackets' state.

        Parameters
        ----------
        tracklets: Iterable[Tracklet]
            Tracklets
        """

        if len(tracklets) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in tracklets])
            multi_covariance = np.asarray([st.covariance for st in tracklets])
            for i, st in enumerate(tracklets):
                if st.state != TrackState.TRACKING:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = kalman_filter.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracklets[i].mean = mean
                tracklets[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame: int, track_id: int):
        """
        Activate the tracklet, assigning to it a `track_id`. Track IDs are
        only assigned once a tracked is activated, because benchmarks often
        limit the maximum amount of track IDs to a value less than 1000.
        """
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = self.kalman_filter.initiate(
            tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.TRACKING
        self.frame_id = frame
        self.start_frame = frame

    def re_activate(self, new_track: Tracklet, frame_id: int):
        assert new_track.state == TrackState.NEW

        if self.use_kalman:
            assert self.kalman_filter is not None

            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, tlwh_to_xyah(new_track.tlwh)
            )
        else:
            self.mean, self.covariance = None, None
            self._tlwh = np.asarray(new_track.tlwh, dtype=np.float64)

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.TRACKING
        self.is_activated = True
        self.frame_id = frame_id
        self.index = new_track.index

    def update(self, new_track: Tracklet, frame: int) -> None:
        """
        Join a track with a track at a later frame. The current `track_id` is
        retrained for propagation.

        Parameters
        ----------
        new_track : Tracklet
            Future tracklet to join with.
        frame : int
            Frame number.
        """
        assert new_track.state == TrackState.NEW

        assert (
            self.frame_id < frame
        ), f"Attempted to join track from frame {self.frame_id} to {frame}."

        self.frame_id = frame
        self.tracklet_len += 1
        self.index = new_track.index

        new_tlwh = new_track.tlwh
        if self.use_kalman:
            assert self.kalman_filter is not None
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, tlwh_to_xyah(new_tlwh)
            )
        else:
            self.mean, self.covariance = None, None
            self._tlwh = np.asarray(new_tlwh, dtype=np.float64)
        self.state = TrackState.TRACKING
        self.is_activated = True
        self.score = new_track.score
        self.category = new_track.category
        self.update_features(new_track.curr_feat)

    @property
    def tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        If a Kalman filter is used, then the mean location is returned.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        return tlwh_to_xyah(self.tlwh)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f" {self.unique_id} [{self.start_frame}:{self.end_frame}]"
        )


def joint_stracks(a: list[Tracklet], b: list[Tracklet]) -> list[Tracklet]:
    exists = {}
    res = []
    for t in a:
        exists[t.unique_id] = 1
        res.append(t)
    for t in b:
        tid = t.unique_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.unique_id] = t
    for t in tlistb:
        tid = t.unique_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, ioudist=0.15):
    pdist = cost.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < ioudist)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):  # type: ignore
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
