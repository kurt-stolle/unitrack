"""
PyTorch implementation of the Hungarian algorithm for solving the assignment problem.
"""

from typing import Tuple

import torch
from torch import Tensor

from .base_assignment import Assignment

__all__ = ["HungarianAssignment"]


class HungarianAssignment(Assignment):
    def forward(self, cost_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        # Step 1: Subtract the minimum cost from each row.
        cost_matrix = self._subtract_min_row(cost_matrix)

        # Step 2: Subtract the minimum cost from each column.
        cost_matrix = self._subtract_min_col(cost_matrix)

        # Step 3: Find the minimum number of lines that cover all the zeros in the matrix.
        lines = self._find_lines(cost_matrix)

        # Step 4: Subtract the minimum uncovered value from each uncovered element in the matrix.
        #         Add it to each element that is covered with two lines.
        uncovered_min = cost_matrix[lines == 0].min()
        cost_matrix[lines == 0] -= uncovered_min
        cost_matrix[lines == 2] += uncovered_min

        # Step 5: If all elements are covered, then stop.
        #         Otherwise, go back to step 3.
        if lines.sum() < cost_matrix.numel():
            return self.forward(cost_matrix)

        # Step 6: Create a set of starred zeros.
        #         Create a set of primed zeros.
        #         For each column, count the number of starred zeros.
        #         If there is exactly one starred zero, then star the primed zero in the same row.
        #         If there are no starred zeros, then prime the first zero in the column.
        #         If there are more than one starred zeros, then do nothing.
        starred, primed = self._find_starred_primed(cost_matrix)
        col_count = torch.bincount(starred.nonzero()[1], minlength=cost_matrix.size(1))
        for i in range(cost_matrix.size(1)):
            if col_count[i] == 1:
                row = starred.nonzero()[0][starred.nonzero()[1] == i]
                primed[row, i] = 1
            elif col_count[i] == 0:
                row = cost_matrix[:, i].nonzero()[0][0]
                primed[row, i] = 1

        # Step 7: Cover all columns with starred zeros.
        #         Find the smallest uncovered value.
        #         Subtract this value from each uncovered element.
        #         Add this value to each element that is covered with two lines.
        col_count = torch.bincount(starred.nonzero()[1], minlength=cost_matrix.size(1))
        uncovered_min = cost_matrix[lines == 0].min()
        cost_matrix[lines == 0] -= uncovered_min
        cost_matrix[lines == 2] += uncovered_min

        # Step 8: Go back to step 3.
        return self.forward(cost_matrix)

    def _subtract_min_row(self, cost_matrix: Tensor) -> Tensor:
        """
        Subtracts the minimum cost from each row.

        Parameters
        ----------
        cost_matrix
            Cost matrix

        Returns
        -------
            Updated cost matrix.
        """
        min_row = cost_matrix.min(dim=1)[0].unsqueeze(1)
        cost_matrix -= min_row.expand_as(cost_matrix)
        return cost_matrix

    def _subtract_min_col(self, cost_matrix: Tensor) -> Tensor:
        """
        Subtracts the minimum cost from each column.

        Parameters
        ----------
        cost_matrix
            Cost matrix

        Returns
        -------
            Updated cost matrix.
        """
        min_col = cost_matrix.min(dim=0)[0].unsqueeze(0)
        cost_matrix -= min_col.expand_as(cost_matrix)
        return cost_matrix

    def _find_lines(self, cost_matrix: Tensor) -> Tensor:
        """
        Finds the minimum number of lines that cover all the zeros in the matrix.

        Parameters
        ----------
        cost_matrix
            Cost matrix

        Returns
        -------
            Matrix with zeros and lines.
        """
        lines = torch.zeros(cost_matrix.size(), dtype=torch.uint8)
        for i, _ in enumerate(cost_matrix):
            row = cost_matrix[i].nonzero()[0]
            # If there is only one zero in the row, then cover this zero and the column.
            if row.numel() == 1:
                lines[i, row] = 1
                lines[:, row] = 1
            # If there are two or more zeros, then cover the first and the second zero.
            elif row.numel() > 1:
                lines[i, row[0]] = 1
                lines[i, row[1]] = 1
        return lines

    def _find_starred_primed(self, cost_matrix: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Finds a set of starred zeros and primed zeros.

        Parameters
        ----------
        cost_matrix
            Cost matrix

        Returns
        -------
            Tuple of starred zeros and primed zeros.
        """
        starred = torch.zeros(cost_matrix.size(), dtype=torch.uint8)
        primed = torch.zeros(cost_matrix.size(), dtype=torch.uint8)

        for i, _ in enumerate(cost_matrix):
            row = cost_matrix[i].nonzero()[0]
            if row.numel() == 1:
                starred[i, row] = 1
            elif row.numel() > 1:
                starred[i, row[0]] = 1
                primed[i, row[1]] = 1
        return starred, primed
