from typing import Dict, Final, List, Optional

import torch

from .detections import Detections
from .states import State
from .states import Value as ValueState
from .tracker import TrackerResult

__all__ = ["Tracklets"]


class Tracklets(torch.nn.Module):
    """
    Container module for tracking predictions/paths, canonically called
    "tracklets" in literature.

    A tracklet is a generic representation of any tracked collection of states,
    e.g. objects in a scene. States of the tracklets are implemented in
    subclasses of :class:`Field`, which may be used to compute a distance
    matrix between tracklets at different frames.
    """

    state_frame: Final = "_frame"
    state_id: Final = "_id"
    state_start: Final = "_start"
    state_active: Final = "_active"

    frame: int
    count: int
    sequence_id: Optional[str]
    max_id: int

    def __init__(self, states: List[State], max_id=1000):
        super().__init__()

        assert max_id > 0, max_id
        assert len(states) > 0, len(states)

        self.max_id = max_id
        self.states = torch.nn.ModuleDict(
            {
                state.id: state
                for state in (
                    ValueState(id=self.state_frame, dtype=torch.int),
                    ValueState(id=self.state_start, dtype=torch.int),
                    ValueState(id=self.state_id, dtype=torch.int),
                    ValueState(id=self.state_active, dtype=torch.bool),
                    *states,
                )
            }
        )

        self.frame = -1
        self.count = 0
        self.sequence_id = None

    def __len__(self) -> int:
        return self.count

    def forward(
        self, sequence_id: str, frame: int, res: TrackerResult
    ) -> torch.Tensor:
        """
        Update the ``Tracklets`` at a given frame with new state data from
        detections.

        Parameters
        ----------
        frame
            Current frame.
        put
            Detections that are put into the current state.
        new
            Detections that are newly added to the state.

        Returns
        -------
            New states.
        """

        assert res.update.indices.dtype == torch.int64, res.update.indices.dtype
        assert res.extend.indices.dtype == torch.int64, res.extend.indices.dtype
        assert res.matches.dtype == torch.int64, res.matches.dtype

        device = res.matches.device

        if not torch.allclose(
            res.update.indices,
            torch.arange(
                len(res.update), device=device, dtype=res.update.indices.dtype
            ),
        ):
            raise ValueError(
                "Updated detections are not in the correct ordering!"
            )
        if not res.update.mutable:
            raise ValueError(
                "Updated observations must be a mutable detections collection!"
            )
        if res.extend.mutable:
            raise ValueError(
                "Updated additions should not be a mutable collection!"
            )

        self.frame = frame
        self.count += 1
        self.sequence_id = sequence_id

        # Pull IDs from update for each detection in the tracker result
        detection_ids = torch.zeros_like(res.matches, dtype=torch.int)

        match_mask = (res.matches >= 0).to(torch.bool)
        match_update_indices = res.matches[match_mask]

        detection_ids[match_mask] = res.update.get(self.state_id)[
            res.update.indices[match_update_indices]
        ]

        # Put new measurements into state
        update_num = len(res.update)
        if update_num > 0:
            for id, state in self.states.items():
                state.update(res.update.get(id))  # type: ignore

        # Add new measurements into states
        extend_num = len(res.extend)
        extend_ids = (
            torch.arange(extend_num, dtype=torch.int, device=device) + 1
        )
        if update_num > 0:
            extend_ids += res.update.get(self.state_id).max()
        if extend_num > 0:
            for id, state in self.states.items():
                if id == self.state_active:
                    state.extend(  # type: ignore
                        torch.as_tensor(
                            [True] * extend_num, dtype=torch.bool, device=device
                        )
                    )
                elif id == self.state_frame:
                    state.extend(  # type: ignore
                        torch.as_tensor(
                            [frame] * extend_num,
                            dtype=torch.int,
                            device=device,
                        )
                    )
                elif id == self.state_start:
                    state.extend(  # type: ignore
                        torch.as_tensor(
                            [frame] * extend_num,
                            dtype=torch.int,
                            device=device,
                        )
                    )
                elif id == self.state_id:
                    state.extend(extend_ids)  # type: ignore
                elif id in res.extend:
                    state.extend(res.extend.get(id))  # type: ignore
                else:
                    raise ValueError(
                        f"State '{id}' does not match a field in {res.extend}!"
                    )

        detection_ids[~match_mask] = extend_ids

        return detection_ids

    def reset(self) -> None:
        """
        Reset the states of this ``Tracklets`` module.
        """
        self.frame = -1
        self.count = 0
        self.sequence_id = None

        for state in self.states.values():
            state.reset()  # type: ignore

    def observe(self) -> Detections:
        """
        Observe the current state of tracklets.

        Returns
        -------
            A ``Detections`` object will all observed states.
        """

        field_values = {}
        for id, state in self.states.items():
            field_values[id] = state.observe()  # type: ignore

        return Detections(field_values, mutable=True)
