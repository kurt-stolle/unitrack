"""

"""
from __future__ import annotations

import typing as T
from typing import Mapping, Optional, cast

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictSequential

from .constants import KEY_ACTIVE, KEY_DELTA, KEY_FRAME, KEY_ID, KEY_INDEX, KEY_START
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
        max_id: int = 1000,
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
            print(f"Adding IDs: {extend_ids}")
        else:
            print(f"This is frame 0, because we have {current_ids.tolist()}")
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

            print(f"State {id} after: {state.memory}")
            print(f"State {id} observed: {state.observe()}")

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
        result_ids[active_indices] = obs.get_at(KEY_ID, active_mask)
        # if active_num > 0:
        #    ids[obs_active.get(KEY_INDEX)] = obs_active.get(KEY_ID)

        extend_indices = new.get(KEY_INDEX)
        result_ids[extend_indices] = extend_ids

        # if extend_num > 0:
        #    ids[new.get(KEY_INDEX)] = extend_ids
        if (result_ids < 0).any():
            msg = f"IDs are not assigned correctly, no negative values should have been propagated! Got: {result_ids}"
            raise ValueError(msg)

        # Update the frame and count
        self.frame.fill_(frame)
        self.count += 1

        return result_ids

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
            {id: state.evolve(delta) for id, state in self.states.items()}
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
        self.frame.fill_(-1)
        self.count.fill_(0)

        for state in self.states.values():
            state.reset()  # type: ignore


def _safe_masked_set_at(
    obs: TensorDictBase, key: str, value: torch.Tensor, mask: torch.Tensor
) -> None:
    if not mask.any():
        return
    obs.set_at_(key, value, mask)
