from typing import Optional, Tuple

import torch

__all__ = ["Auction"]


from .base_assignment import Assignment


class Auction(Assignment):
    """
    Solves the linear assignment over a cost matrix using an auction algorithm.
    """

    bid_size: torch.Tensor

    def __init__(self, bid_size=1.0, threshold=1.0):
        """
        Parameters
        ----------
        bid_size, optional
            Step size of auction bids, by default None
        """
        super().__init__(threshold=threshold)

        self.register_buffer("bid_size", torch.as_tensor(bid_size))

    def forward(self, cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = self.bid_size / min(cost_matrix.shape)

        device = cost_matrix.device

        cost = torch.zeros((1, cost_matrix.shape[1]), device=device)
        ass = torch.full((cost_matrix.shape[0],), -1, device=device, dtype=torch.long)
        bids = torch.zeros_like(cost_matrix)

        counter = 0
        while (ass == -1).any():
            counter += 1

            # --
            # Bidding

            unassigned = (ass == -1).nonzero().squeeze()

            value = cost_matrix[unassigned] - cost
            top_value, top_idx = value.topk(2, dim=1)

            first_idx = top_idx[:, 0]
            first_value, second_value = top_value[:, 0], top_value[:, 1]

            bid_increments = first_value - second_value + eps

            bids_ = bids[unassigned]
            bids_.zero_()
            if unassigned.dim() == 0:
                high_bids, high_bidders = bid_increments, unassigned
                cost[:, first_idx] += high_bids
                ass[(ass == first_idx).nonzero()] = -1
                ass[high_bidders] = first_idx
            else:
                bids_.scatter_(
                    dim=1,
                    index=first_idx.contiguous().view(-1, 1),
                    src=bid_increments.view(-1, 1),
                )

                have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

                high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)

                high_bidders = unassigned[high_bidders.squeeze()]

                cost[:, have_bidder] += high_bids

                ind = (ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1).nonzero()
                ass[ind] = -1

                ass[high_bidders] = have_bidder.squeeze()

        score = cost_matrix.gather(dim=1, index=ass.view(-1, 1)).sum()

        return score, ass
