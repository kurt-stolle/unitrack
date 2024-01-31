"""
PyTorch implementation of the Hungarian algorithm for solving the assignment problem.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from .base_assignment import Assignment

__all__ = ["Hungarian", "hungarian_assignment"]


class Hungarian(Assignment):
    def _assign(self, cost_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Solves the assignment problem using the Hungarian algorithm.

        Parameters
        ----------
        cost_matrix
            Cost matrix

        Returns
        -------
            Tuple of the optimal assignment and the total assignment cost.
        """
        return hungarian_assignment(cost_matrix)


@torch.jit.script_if_tracing
@torch.no_grad()
def hungarian_assignment(cost_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    N, M = cost_matrix.shape

    # Step 1: Subtract the minimum value from each row
    cost_matrix -= cost_matrix.min(dim=1, keepdim=True).values

    # Step 2: Subtract the minimum value from each column
    cost_matrix -= cost_matrix.min(dim=0, keepdim=True).values

    mask = torch.zeros_like(cost_matrix, dtype=torch.bool)
    row_mask = torch.ones(N, dtype=torch.bool)
    col_mask = torch.ones(M, dtype=torch.bool)

    while True:
        # Step 3: Find zeros that have no other zeros in the same row or column
        mask[row_mask] = cost_matrix[row_mask] == 0
        mask[:, col_mask] = mask[:, col_mask] * (cost_matrix[:, col_mask] == 0)

        # Step 4: If there are enough zeros, terminate
        if mask.sum() >= min(N, M):
            break

        # Step 5: Otherwise, update cost matrix by adding the minimum uncovered value
        min_val = cost_matrix[row_mask][:, col_mask].min()
        cost_matrix[row_mask] -= min_val
        cost_matrix[:, ~col_mask] += min_val

    # Extract matched row and column indices
    rows, cols = torch.where(mask)
    matches = torch.stack((rows, cols), dim=1)

    # Identify unmatched rows and columns
    unmatched_rows = torch.tensor(
        [i for i in range(N) if not (rows == i).any()], device=cost_matrix.device
    )
    unmatched_cols = torch.tensor(
        [j for j in range(M) if not (cols == j).any()], device=cost_matrix.device
    )

    return matches, unmatched_rows, unmatched_cols
