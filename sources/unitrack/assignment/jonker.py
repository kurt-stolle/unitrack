from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from lap import lapjv

from .base_assignment import Assignment

__all__ = ["Jonker", "jonker_volgenant_assignment"]


class Jonker(Assignment):
    def _assign(
        self, cost_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return jonker_volgenant_assignment(cost_matrix)


@torch.jit.ignore()
@torch.no_grad()
def jonker_volgenant_assignment(
    cost_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform linear assignment. If possible, an assignment on the diagonal of the
    matrix is preferred if this assignment has equal cost to the algorithm
    result.

    TODO: PyTorch implementation
    """

    device = cost_matrix.device
    cost_matrix = cost_matrix.detach().cpu().numpy()

    matches, unmatched_a, unmatched_b = [], [], []

    # Jonker algorithm, i.e. linear sum assignment (rows) -> (cols)
    cost, x, y = lapjv(cost_matrix, extend_cost=True)

    # Create match and unassigned lists
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = torch.from_numpy(np.where(x < 0)[0]).long()
    unmatched_b = torch.from_numpy(np.where(y < 0)[0]).long()
    matches = torch.from_numpy(np.asarray(matches)).long()

    return matches.to(device), unmatched_a.to(device), unmatched_b.to(device)
