"""
PyTorch implementation of the Hungarian algorithm for solving the assignment problem.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.fx
import typing_extensions as TX
from torch import Tensor

from ._base import Assignment

__all__ = ["Hungarian", "hungarian_assignment"]


class Hungarian(Assignment):
    r"""
    Implements the Hungarian algorithm for solving a linear assignment problem.
    """

    @TX.override
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


def hungarian_assignment(
    cost_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform linear assingment using the SciPy implementation
    """

    from scipy.optimize import linear_sum_assignment

    cm = cost_matrix.cpu().numpy()
    cm = np.ascontiguousarray(cm).astype(np.float64)
    cm = np.where(np.isfinite(cm), cm, np.inf)

    row_ind, col_ind = linear_sum_assignment(cm)

    matches = torch.from_numpy(np.column_stack((row_ind, col_ind))).clone().long()
    unmatched_a = (
        torch.from_numpy(np.setdiff1d(np.arange(cm.shape[0]), row_ind)).clone().long()
    )
    unmatched_b = (
        torch.from_numpy(np.setdiff1d(np.arange(cm.shape[1]), col_ind)).clone().long()
    )

    return (
        matches.to(cost_matrix.device),
        unmatched_a.to(cost_matrix.device),
        unmatched_b.to(cost_matrix.device),
    )


torch.fx.wrap("hungarian_assignment")

# @torch.jit.script_if_tracing
# @torch.no_grad()
# def hungarian_assignment(cost_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
#     N, M = cost_matrix.shape

#     # Step 1: Subtract the minimum value from each row
#     cost_matrix -= cost_matrix.min(dim=1, keepdim=True).values

#     # Step 2: Subtract the minimum value from each column
#     cost_matrix -= cost_matrix.min(dim=0, keepdim=True).values

#     mask = torch.zeros_like(cost_matrix, dtype=torch.bool)
#     row_mask = torch.ones(N, dtype=torch.bool)
#     col_mask = torch.ones(M, dtype=torch.bool)

#     while True:
#         # Step 3: Find zeros that have no other zeros in the same row or column
#         mask[row_mask] = cost_matrix[row_mask] == 0
#         mask[:, col_mask] = mask[:, col_mask] * (cost_matrix[:, col_mask] == 0)

#         # Step 4: If there are enough zeros, terminate
#         if mask.sum() >= min(N, M):
#             break

#         # Step 5: Otherwise, update cost matrix by adding the minimum uncovered value
#         min_val = cost_matrix[row_mask][:, col_mask].min()
#         cost_matrix[row_mask] -= min_val
#         cost_matrix[:, ~col_mask] += min_val

#     # Extract matched row and column indices
#     rows, cols = torch.where(mask)
#     matches = torch.stack((rows, cols), dim=1)

#     # Identify unmatched rows and columns
#     unmatched_rows = torch.tensor(
#         [i for i in range(N) if not (rows == i).any()], device=cost_matrix.device
#     )
#     unmatched_cols = torch.tensor(
#         [j for j in range(M) if not (cols == j).any()], device=cost_matrix.device
#     )

#     return matches, unmatched_rows, unmatched_cols
