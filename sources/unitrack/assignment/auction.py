from typing import Tuple

import torch

from .base_assignment import Assignment

__all__ = ["Auction", "auction_assignment"]


class Auction(Assignment):
    """
    Solves the linear assignment over a cost matrix using an auction algorithm.
    """

    bid_size: torch.jit.Final[float]

    def __init__(self, bid_size=0.05, *args, **kwargs):
        """
        Parameters
        ----------
        bid_size, optional
            Step size of auction bids, by default None
        """
        super().__init__(*args, **kwargs)

        self.bid_size = bid_size

    def _assign(self, cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return auction_assignment(cost_matrix, self.bid_size)


@torch.jit.script_if_tracing
@torch.no_grad()
def auction_assignment(cost_matrix: torch.Tensor, bid_size: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute epsilon based on bid_size and the smaller dimension of the cost matrix
    eps = bid_size / min(cost_matrix.shape)

    device = cost_matrix.device

    # Initialize cost, assignments, and bids tensors
    cost = torch.zeros((1, cost_matrix.shape[1]), device=device)
    ass = torch.full((cost_matrix.shape[0],), -1, device=device, dtype=torch.long)
    bids = torch.zeros_like(cost_matrix)

    # Iterate until all rows are assigned
    while (ass < 0).sum() > max(0, cost_matrix.shape[0] - cost_matrix.shape[1]):
        # Get indices of unassigned rows
        unassigned = (ass == -1).nonzero().flatten()

        # Calculate value matrix for unassigned rows
        value = cost_matrix[unassigned] - cost

        # Get top 2 values and their indices for each unassigned row
        top_value, top_idx = torch.where(value.isfinite(), value, 0).topk(2, dim=1)

        # Calculate bid increments for the highest values
        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]

        bid_increments = first_value - second_value + eps

        # Reset bids for unassigned rows
        bids.index_fill_(0, unassigned, 0)
        # bids.zero_()
        # Update bids for the highest values
        bids[unassigned] = bids[unassigned].scatter_(dim=1, index=first_idx.view(-1, 1), src=bid_increments.view(-1, 1))

        # Get columns with bidders
        have_bidder = (bids > 0).int().sum(dim=0).nonzero().squeeze()

        # Get maximum bids and their row indices
        high_bids, high_bidders = bids[:, have_bidder].max(dim=0)
        # high_bidders = unassigned[high_bidders.squeeze()]
        high_bidders = high_bidders.squeeze()

        # Update cost matrix with the high bids
        cost[:, have_bidder] += high_bids

        # Temporarily unassign rows that were previously assigned to the current winning columns
        ass[(ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1).nonzero().squeeze()] = -1

        # Assign high bidders to the winning columns
        ass[high_bidders] = have_bidder.squeeze()

        # Set matched rows in the cost matrix to infinity
        # cost_matrix[high_bidders, :] = torch.inf
        # cost_matrix[:, have_bidder] = torch.inf

    # Get matches, unmatched_rows, and unmatched_cols
    idx = torch.arange(ass.numel())
    matches = torch.stack([idx, ass], dim=1)
    matches = matches[ass >= 0]
    unmatched_rows = ass[ass < 0]
    unmatched_cols = (torch.arange(cost_matrix.shape[1])[None, :] != ass[:, None]).all(dim=0).nonzero().squeeze()

    return matches, unmatched_rows, unmatched_cols
