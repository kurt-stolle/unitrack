from typing import Mapping

import torch
from tensordict import TensorDict, TensorDictBase

from .constants import KEY_ACTIVE, KEY_FRAME, KEY_ID, KEY_INDEX, KEY_START
from .context import Context
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

    def __init__(self, states: Mapping[str, State], max_id=1000):
        super().__init__()

        assert max_id > 0, max_id
        assert len(states) > 0, len(states)

        self.max_id = max_id
        self.states = torch.nn.ModuleDict()  # type: ignore
        self.states[KEY_FRAME] = ValueState(torch.int)
        self.states[KEY_START] = ValueState(torch.int)
        self.states[KEY_ID] = ValueState(torch.int)
        self.states[KEY_ACTIVE] = ValueState(torch.bool)
        self.states.update(states)

        self.register_buffer("frame", torch.tensor(-1, dtype=torch.int, requires_grad=False))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int, requires_grad=False))

    def __len__(self) -> int:
        return int(self.count.detach().cpu().item())

    def forward(self) -> None:
        raise NotImplementedError("TrackletMemory does not implement forward()!")

    @torch.jit.export
    def update(self, ctx: Context, obs: TensorDictBase, new: TensorDictBase) -> torch.Tensor:
        """
        Apply the updated observations and new detections to the tracklets states.

        Returns the complete set of IDs of the context (which is also updated in-place).
        """

        self.frame.fill_(ctx.frame)
        self.count += 1

        extend_num = new.batch_size[0]
        extend_ids = torch.arange(extend_num, dtype=torch.long, device=self.frame.device, requires_grad=False)
        extend_ids += 1
        if len(obs.get(KEY_ID)) > 0:
            extend_ids += obs.get(KEY_ID).max()

        ctx.ids[new.get(KEY_INDEX)] = extend_ids

        extend_frame = torch.full(
            (extend_num,),
            ctx.frame,
            dtype=torch.int,
            device=self.frame.device,
        )

        assert extend_ids.max() < self.max_id, (extend_ids.max().cpu().item(), self.max_id)

        for id, state in self.states.items():
            state.update(obs.get(id))

            if id == KEY_ACTIVE:
                state.extend(torch.ones((extend_num,), dtype=torch.bool, device=self.frame.device, requires_grad=False))  # type: ignore
            elif id == KEY_FRAME or id == KEY_START:
                state.extend(extend_frame)
            elif id == KEY_ID:
                state.extend(extend_ids)  # type: ignore
            elif id in new.keys():
                state.extend(new.get(id))  # type: ignore
            else:
                raise ValueError(f"State '{id}' does not match a field in {new.keys()}!")

        return ctx.ids

    @torch.jit.export
    def reset(self) -> None:
        """
        Reset the states of this ``Tracklets`` module.
        """
        self.frame.fill_(-1)
        self.count.fill_(0)

        for state in self.states.values():
            state.reset()  # type: ignore

    @torch.jit.export
    def observe(self) -> TensorDictBase:
        """
        Observe the current state of tracklets.

        Returns
        -------
            A ``TensorDict`` object will all observed states.
        """

        obs = {}
        for id, state in self.states.items():
            obs[id] = state.observe()  # type: ignore

        return TensorDict.from_dict(obs, device=self.frame.device)
