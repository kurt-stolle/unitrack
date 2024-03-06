"""

"""

from __future__ import annotations

import typing as T
from typing import Mapping, Optional, cast

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictSequential

from .constants import (
    DEBUG,
    KEY_ACTIVE,
    KEY_DELTA,
    KEY_FRAME,
    KEY_ID,
    KEY_INDEX,
    KEY_START,
)
from .states import State
from .states import Value as ValueState

__all__ = ["TrackletMemory"]


class TrackletMemory(torch.nn.Module):
    """
    Container module for tracking predictions/paths, canonically called
    "tracklets" in literature.

    A tracklet is a generic representation of any tracked collection of states,
    e.g. objects in a scene. States of the tracklets are implemented in
    subclasses of :class:`Field`, which may be used to compute a distance
    matrix between tracklets at different frames.
    """

    frame: torch.Tensor
    count: torch.Tensor
    max_id: torch.jit.Final[int]
    states: Mapping[str, State]

    def __init__(
        self,
        states: Mapping[str, State],
        max_id: int = 2**15,
        auto_reset: bool = True,
        fps: int = 15,
    ):
        super().__init__()

        assert max_id > 0, max_id
        assert len(states) > 0, len(states)

        self.fps = fps
        self.max_id = max_id
        self.states = torch.nn.ModuleDict()  # type: ignore
        self.states[KEY_FRAME] = ValueState(torch.int64)
        self.states[KEY_START] = ValueState(torch.int64)
        self.states[KEY_ID] = ValueState(torch.int64)
        self.states[KEY_ACTIVE] = ValueState(torch.bool)
        self.states.update(states)

        self.auto_reset = auto_reset  # reset the memory when the current frame is larger than the stored frame

        self.register_buffer(
            "frame",
            torch.tensor(-1, dtype=torch.int, requires_grad=False),
            persistent=False,
        )
        self.register_buffer(
            "count",
            torch.tensor(0, dtype=torch.int, requires_grad=False),
            persistent=False,
        )

    def __len__(self) -> int:
        return int(self.count.detach().cpu().item())

    @torch.jit.export
    def write(
        self, ctx: TensorDictBase, obs: TensorDictBase, new: TensorDictBase
    ) -> torch.Tensor:
        """
        Apply the updated observations and new detections to the tracklets states.

        Returns the complete set of IDs of the context (which is also updated in-place).
        """

        frame = ctx.get(KEY_FRAME)

        # Update observations' states
        active_mask = obs.get(KEY_ACTIVE)
        # obs_active = obs.get_sub_tensordict(obs.get(KEY_ACTIVE))
        # obs_active = obs[obs.get(KEY_ACTIVE)]
        active_num = active_mask.to(torch.int).sum()
        # if active_num > 0:
        #    obs_active = obs_active.set(
        #        KEY_FRAME,
        #        frame.expand(obs_active.batch_size[:1]),
        #    )
        obs.set_at_(KEY_FRAME, frame.to(obs.device), active_mask)

        # Read the amount of added instances from the batch size of the new detections TensorDict.
        extend_num = new.batch_size[0]
        total_num = active_num + extend_num

        # Assign IDs to the new detections.
        extend_ids = torch.arange(
            1,
            extend_num + 1,
            dtype=torch.long,
            device=self.frame.device,
            requires_grad=False,
        )
        current_ids = obs.get(KEY_ID)
        if len(current_ids) > 0:
            extend_ids += current_ids.max()

        if DEBUG:
            print(f"Current IDs: {current_ids}")
            print(f"Extend IDs: {extend_ids}")

        if extend_num > 0 and extend_ids.max() >= self.max_id:
            msg = (
                f"Most recent ID {extend_ids.max()} has a value larger than the "
                f"maximum allowed ID {self.max_id}!"
            )
            raise RuntimeError(msg)

        # Update and/or extend the states.
        for id, state in self.states.items():
            # State.foward = (update, extend) -> None.
            state_upd: torch.Tensor = obs.get(id)
            state.update(state_upd)

            if extend_num > 0:
                state_ext: torch.Tensor

                # Switch per special state key.
                if id == KEY_ACTIVE:
                    state_ext = torch.ones(
                        (extend_num,),
                        dtype=torch.bool,
                        device=self.frame.device,
                        requires_grad=False,
                    )
                elif id in (KEY_FRAME, KEY_START):
                    state_ext = torch.full(
                        (extend_num,),
                        fill_value=frame,
                        dtype=torch.int,
                        device=self.frame.device,
                    )
                elif id == KEY_ID:
                    state_ext = extend_ids
                elif id in T.cast(T.Iterable[int], new.keys()):
                    state_ext = new.get(id)
                else:
                    raise ValueError(
                        f"State '{id}' does not match a field in {new.keys()}!"
                    )
                state.extend(state_ext)

        # Create the return value: a list of IDs to assign to detections in the next
        # frame.
        result_ids = torch.full(
            (total_num,),
            -1,
            dtype=torch.long,
            device=self.frame.device,
            requires_grad=False,
        )
        active_indices = obs.get_at(KEY_INDEX, active_mask)
        assert (
            active_indices >= 0
        ).all(), f"Active indices are not correct! Got: {active_indices.tolist()}"
        result_ids[active_indices] = active_ids = obs.get_at(
            KEY_ID, active_mask
        ).clone()

        if DEBUG:
            print("Propagated IDs: ")
            for _idx, _id in zip(active_indices, active_ids):
                print(f"   {_id} -> detection {_idx}")

        # if active_num > 0:
        #    ids[obs_active.get(KEY_INDEX)] = obs_active.get(KEY_ID)

        extend_indices = new.get(KEY_INDEX)
        assert not (extend_indices < 0).any(), (
            "New detections have negative indices! " f"Got: {extend_indices.tolist()}"
        )
        assert not (extend_indices > total_num).any(), (
            "New detections have indices larger than the total amount of detections! "
            f"Got: {extend_indices.tolist()}"
        )

        assert not torch.isin(extend_indices, active_indices).any(), (
            "New detections have the same index as active detections! "
            f"Got: {extend_indices.tolist()}"
        )
        result_ids[extend_indices] = extend_ids

        # if extend_num > 0:
        #    ids[new.get(KEY_INDEX)] = extend_ids
        if (result_ids < 0).any():
            msg = f"IDs are not assigned correctly, no negative values should have been propagated! Got: {result_ids}"
            raise ValueError(msg)

        # Update the frame and count
        self.frame.fill_(frame)
        self.count += 1

        return result_ids.contiguous()

    @torch.jit.export
    def read(self, frame: int) -> tuple[TensorDictBase, TensorDictBase]:
        """
        Observe the current state of tracklets. Evolves the states of the tracklets
        to the current frame.

        Parameters
        ----------
        frame
            The current frame number, used to automatically reset the memory before
            reading when ``auto_reset`` is enabled.

        Returns
        -------
            A ``TensorDict`` object will all observed states.
        """

        if self.frame >= frame:
            if self.auto_reset:
                # Reset because the current frame is larger than the stored frame.
                self.reset()
            else:
                raise ValueError(
                    f"Cannot read frame {frame} from memory at frame {self.frame}!"
                )

        # Calculate the time-delta from the amoutn of frames passed and the FPS.
        delta = (frame - self.frame) / self.fps

        ctx = TensorDict.from_dict(
            {
                KEY_DELTA: delta,
                KEY_FRAME: frame,
            },
            batch_size=[],
            device=self.frame.device,
        )

        obs = TensorDict.from_dict(
            {id: state.evolve(delta) for id, state in self.states.items()},
            batch_dims=1,
            device=self.frame.device,
        )
        obs[KEY_INDEX] = torch.full(
            obs.batch_size[:1],
            -1,
            dtype=torch.long,
            device=self.frame.device,
            requires_grad=False,
        )

        return ctx, obs

    @torch.jit.export
    def reset(self) -> None:
        """
        Reset the states of this ``Tracklets`` module.
        """

        if DEBUG:
            print("Resetting memory")
        self.frame.fill_(-1)
        self.count.fill_(0)

        for state in self.states.values():
            state.reset()  # type: ignore
