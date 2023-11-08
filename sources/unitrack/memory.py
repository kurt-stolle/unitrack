from typing import Mapping, Optional, cast

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictSequential

from .constants import (
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

    def __init__(self, states: Mapping[str, State], max_id=1000, auto_reset=True, fps=15):
        super().__init__()

        assert max_id > 0, max_id
        assert len(states) > 0, len(states)

        self.fps = fps
        self.max_id = max_id
        self.states = torch.nn.ModuleDict()  # type: ignore
        self.states[KEY_FRAME] = ValueState(torch.int)
        self.states[KEY_START] = ValueState(torch.int)
        self.states[KEY_ID] = ValueState(torch.int)
        self.states[KEY_ACTIVE] = ValueState(torch.bool)
        self.states.update(states)

        self.auto_reset = auto_reset  # reset the memory when the current frame is larger than the stored frame

        self.register_buffer("frame", torch.tensor(-1, dtype=torch.int, requires_grad=False))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int, requires_grad=False))

    def __len__(self) -> int:
        return int(self.count.detach().cpu().item())

    @torch.jit.export
    def write(self, ctx: TensorDictBase, obs: TensorDictBase, new: TensorDictBase) -> torch.Tensor:
        """
        Apply the updated observations and new detections to the tracklets states.

        Returns the complete set of IDs of the context (which is also updated in-place).
        """

        frame = ctx.get(KEY_FRAME)

        # Update observations' states
        obs_active = obs.get_sub_tensordict(obs.get(KEY_ACTIVE))
        active_num = obs_active.batch_size[0]
        if active_num > 0:
            obs_active.set_(KEY_FRAME, frame)

        # Read the amount of added instances from the batch size of the new detections TensorDict.
        extend_num = new.batch_size[0]
        total_num = active_num + extend_num

        # Assign IDs to the new detections.
        extend_ids = torch.arange(extend_num, dtype=torch.long, device=self.frame.device, requires_grad=False)
        extend_ids += 1
        if len(obs.get(KEY_ID)) > 0:
            extend_ids += obs.get(KEY_ID).max()
        assert isinstance(extend_ids, torch.Tensor), type(extend_ids)
        assert extend_ids.max() < self.max_id, (extend_ids.max().cpu().item(), self.max_id)

        # Update and/or extend the states.
        for id, state in self.states.items():
            # State.foward = (update, extend) -> None.
            state_upd: torch.Tensor = obs.get(id)
            state_ext: torch.Tensor

            # Switch per special state key.
            if id == KEY_ACTIVE:
                state_ext = torch.ones((extend_num,), dtype=torch.bool, device=self.frame.device, requires_grad=False)
            elif id == KEY_FRAME or id == KEY_START:
                state_ext = torch.full(
                    (extend_num,),
                    frame,
                    dtype=torch.int,
                    device=self.frame.device,
                )
            elif id == KEY_ID:
                state_ext = extend_ids
            elif id in new.keys():
                state_ext = new.get(id)
            else:
                raise ValueError(f"State '{id}' does not match a field in {new.keys()}!")

            # Propagate
            state.update(state_upd)
            state.extend(state_ext)

        # Update the IDs of the new detections (that hadn't been matched)
        ids = torch.full((total_num,), -1, dtype=torch.long, device=self.frame.device, requires_grad=False)
        if active_num > 0:
            ids[obs_active.get(KEY_INDEX)] = obs_active.get(KEY_ID)
        if extend_num > 0:
            ids[new.get(KEY_INDEX)] = extend_ids
        if (ids < 0).any():
            raise ValueError(f"IDs are not assigned correctly: {ids}!")

        # Update the frame and count
        self.frame.fill_(frame)
        self.count += 1

        return ids

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

        if frame >= self.frame:
            if self.auto_reset:
                self.reset()
            else:
                raise ValueError(f"Cannot read frame {frame} from memory at frame {self.frame}!")

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

        obs = TensorDict.from_dict({id: state.evolve(delta) for id, state in self.states.items()})
        obs[KEY_INDEX] = torch.full(
            obs.batch_size[:1], -1, dtype=torch.long, device=self.frame.device, requires_grad=False
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
