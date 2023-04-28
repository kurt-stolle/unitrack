"""
Greedy assignment is a simple assignment algorithm that greedily assigns 
detections to tracklets, by selecting the best match at each step. This
algorithm is not guaranteed to find the optimal solution, but it is fast
and simple to implement.
"""

from typing import Tuple

import torch

from .base_assignment import Assignment


class GreedyAssignment(Assignment):
    def forward(self, cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assigns detections to tracklets using a greedy algorithm. This algorithm
        is not guaranteed to find the optimal solution, but it is fast and
        simple to implement.

        Parameters
        ----------
        cost_matrix
            Cost matrix (NxM) to solve

        Returns
        -------
            Tuple of matches (N_match x M_match), unmatched columns and
            unmatched rows
        """

        num_rows, num_cols = cost_matrix.shape

        unmatched_rows = torch.arange(num_rows, device=cost_matrix.device)
        unmatched_cols = torch.arange(num_cols, device=cost_matrix.device)

        if cost_matrix.numel() == 0:
            return (
                torch.zeros((0, 2), dtype=torch.long, device=cost_matrix.device),
                unmatched_rows,
                unmatched_cols,
            )

        # Greedy assignment
        matches = []
        while unmatched_rows.numel() > 0 and unmatched_cols.numel() > 0:
            # Find minimum cost match
            row_idx = torch.argmin(cost_matrix[unmatched_rows[:, None], unmatched_cols], dim=1)
            col_idx = torch.arange(unmatched_cols.shape[0], device=cost_matrix.device)[row_idx]

            # Add match
            matches.append(torch.stack([unmatched_rows[row_idx], unmatched_cols[col_idx]], dim=1))

            # Remove matched rows and columns
            unmatched_rows = torch.cat([unmatched_rows[:row_idx], unmatched_rows[row_idx + 1 :]])
            unmatched_cols = torch.cat([unmatched_cols[:col_idx], unmatched_cols[col_idx + 1 :]])

        return torch.cat(matches, dim=0), unmatched_rows, unmatched_cols
