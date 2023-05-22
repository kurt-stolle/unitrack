from typing import Dict, List, Mapping, Tuple, cast

import tensordict
import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential

from .constants import KEY_ACTIVE, KEY_FRAME, KEY_ID, KEY_INDEX, KEY_START
from .fields import Field
from .stages import Stage


class MultiStageTracker(nn.Module):
    """
    Multi-stage tracker that applies a cascade of stages to a set of detections.
    """

    def __init__(self, fields: Mapping[str, Field], stages: List[Stage]):
        super().__init__()

        assert len(stages) > 0

        self.fields = TensorDictSequential(
            *(
                TensorDictModule(field, in_keys=[("ctx", , "det"], out_keys=[out_key])
                for out_key, field in fields.items()
            )
        )

        self.stages = cast(List[Stage], nn.ModuleList(stages))

    def forward(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        det: TensorDictBase,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """
        Perform tracking, returns a tuple of updated observations and the field-values of new tracklets.


        Returns
        -------
        Tuple[TensorDict, TensorDict]
            Updated observations and field-values of new tracklets.
        """

        # Read the amount of detections
        num_det = det.batch_size[0]

        # Reset the observations' index field for back-translation
        # if KEY_INDEX in obs:
        #     obs.fill_(KEY_INDEX, -1)
        # else:
        #     obs[KEY_INDEX] = torch.full(
        #         (obs.batch_size[0],),
        #         -1,
        #         dtype=torch.long,
        #         device=obs.device,
        #     )

        # Fields target (newly detected objects)
        new = TensorDict.from_dict(
            {
                KEY_INDEX: torch.arange(num_det, device=det.device),
            }
        )

        # The `new` tensordict is updated in-place, hence why it is preallocated.
        # The result of this function is a tuple of fields in order of the `fields` dict.
        self.fields(ctx=ctx, det=det, tensordict_out=new)

        obs_candidates = obs.get_sub_tensordict(obs.get(KEY_ACTIVE))
        for stage in self.stages:
            obs_candidates, new = stage(ctx, obs_candidates, new)

            from tensordict import SubTensorDict

            assert isinstance(obs_candidates, SubTensorDict), type(obs_candidates)
            assert isinstance(new, SubTensorDict), type(new)

        if len(obs_candidates) > 0 and obs_candidates.batch_size[0] > 0:
            obs_candidates.fill_(KEY_ACTIVE, False)

        return obs, new
