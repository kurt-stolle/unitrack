from typing import Tuple

import torch


class Assignment(torch.nn.Module):
    """
    Solves a linear assignment problem (LAP).
    """

    threshold: float

    def __init__(self, threshold: float = 1.0):
        super().__init__()

        self.threshold = threshold

    def forward(
        self, cost_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve the cost matrix

        Parameters
        ----------
        cost_matrix
            Cost matrix (NxM) to solve

        Returns
        -------
            Tuple of matches (N_match x M_match), unmatched columns and
            unmatched rows
        """
        raise NotImplementedError
