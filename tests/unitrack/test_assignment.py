import pytest
import torch

from unitrack import assignment


@pytest.fixture(
    params=[
        (8, 8),
        (8, 10),
        (10, 8),
        (0, 0),
    ],
    ids=(
        "cost:square",
        "cost:tall",
        "cost:wide",
        "cost:empty",
    ),
)
def cost_matrix(request):
    shape = request.param
    return torch.rand(shape, dtype=torch.float)


@pytest.fixture(
    params=[
        assignment.Greedy,
        assignment.Hungarian,
        assignment.Auction,
        assignment.Jonker,
    ],
    ids=(
        "alg:greedy",
        "alg:hungarian",
        "alg:auction",
        "alg:jonker",
    ),
    scope="module",
)
def solver(request):
    mod = request.param()
    assert isinstance(mod, assignment.Assignment)
    return mod


def test_assignment_invoke(cost_matrix, solver):
    matches, unmatch_rows, unmatch_cols = solver(cost_matrix)

    assert matches.shape[0] <= min(cost_matrix.shape)
    assert matches.shape[1] == 2
    assert unmatch_rows.shape[0] <= cost_matrix.shape[0]
    assert unmatch_cols.shape[0] <= cost_matrix.shape[1]
    assert not any(matches[:, 0] < 0)
    assert not any(matches[:, 1] < 0)
    assert not any(r in matches[:, 0] for r in unmatch_rows)
    assert not any(c in matches[:, 1] for c in unmatch_cols)


@pytest.mark.parametrize(
    "known_cost, known_matches",
    [
        (torch.arange(9, dtype=torch.float).reshape(3, 3), torch.tensor([[0, 0], [1, 1], [2, 2]])),
        (torch.arange(6, dtype=torch.float).reshape(2, 3), torch.tensor([[0, 0], [1, 1]])),
        (torch.arange(6, 0, -1, dtype=torch.float).reshape(2, 3), torch.tensor([[0, 2], [1, 1]])),
    ],
)
def test_assignment_known(known_cost, known_matches, solver):
    """
    Test if all algorithms can solve the known N x M cost matrix to known N x 2 matches
    """

    known_cost = torch.as_tensor(known_cost)
    known_matches = torch.as_tensor(known_matches)

    matches, unmatch_col, unmatch_row = solver(known_cost)

    assert matches.shape == known_matches.shape

    matches = matches[matches[:, 0].argsort(dim=0).flatten(), :]

    assert known_cost[matches[:, 0], matches[:, 1]].sum() <= known_cost[known_matches[:, 0], known_matches[:, 1]].sum()
