from typing import Tuple

import numpy as np
import torch
from lap import lapjv

from ._assignment import Assignment

__all__ = ["Jonker"]


class Jonker(Assignment):
    def forward(
        self, cost_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.linear_assignment(cost_matrix)

    @torch.jit.ignore
    def linear_assignment(
        self, cost_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform linear assignment. If possible, an assignment on the diagonal of the
        matrix is preferred if this assignment has equal cost to the algorithm
        result.

        TODO: PyTorch implementation
        """

        cost_matrix = cost_matrix.detach().cpu().numpy()
        cs_num, ds_num = cost_matrix.shape

        # Skip when cost_matrix has 0-dimension
        if cost_matrix.size == 0:
            return (
                torch.empty((0, 2), dtype=torch.long),
                torch.arange(cs_num, dtype=torch.long),
                torch.arange(ds_num, dtype=torch.long),
            )
        matches, unmatched_a, unmatched_b = [], [], []

        # Jonker algorithm, i.e. linear sum assignment (rows) -> (cols)
        cost, x, y = lapjv(
            cost_matrix, extend_cost=True, cost_limit=self.threshold
        )

        # Create match and unassigned lists
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])

        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)

        return (
            torch.from_numpy(matches).long(),
            torch.from_numpy(unmatched_a).long(),
            torch.from_numpy(unmatched_b).long(),
        )
