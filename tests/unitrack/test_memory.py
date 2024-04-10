r"""
Tests for ``unitrack.memory``.
"""
import typing as T
import unitrack as ut
import torch
import pytest
from tensordict import TensorDict


KEY_STATE_FLOAT: T.Final = "float_state"
KEY_STATE_BOOL: T.Final = "bool_state"
KEY_STATE_INT: T.Final = "int_state"

STATES_USER: T.Final = [KEY_STATE_FLOAT, KEY_STATE_BOOL, KEY_STATE_INT]
STATES_INTERNAL: T.Final = [
    ut.consts.KEY_FRAME,
    ut.consts.KEY_START,
    ut.consts.KEY_ID,
    ut.consts.KEY_ACTIVE,
]


@pytest.fixture
def memory() -> ut.TrackletMemory:
    mem = ut.TrackletMemory(
        states={
            KEY_STATE_FLOAT: ut.states.Value(torch.float32),
            KEY_STATE_BOOL: ut.states.Value(torch.bool),
            KEY_STATE_INT: ut.states.Value(torch.int),
        },
        auto_reset=False,
    )

    assert len(mem) == 0, "Memory should be empty at initialization."
    return mem


@pytest.fixture
def obs_old() -> TensorDict:
    r"""
    A TensorDict representing the trackets at the previous frame.
    """
    return TensorDict(
        {
            KEY_STATE_FLOAT: torch.tensor([1.0, 2.0, 3.0]),
            KEY_STATE_BOOL: torch.tensor([True, False, True]),
            KEY_STATE_INT: torch.tensor([1, 2, 3]),
        },
        batch_size=[3],
    )


@pytest.fixture
def obs_new(obs_old) -> TensorDict:
    r"""
    A TensorDict representing the tracklets at the current frame.
    """
    return TensorDict(
        {
            KEY_STATE_FLOAT: torch.tensor([4.0, 5.0, 6.0]),
            KEY_STATE_BOOL: torch.tensor([False, True, False]),
            KEY_STATE_INT: torch.tensor([4, 5, 6]),
        },
        batch_size=[3],
    )


def test_memory_state(memory: ut.TrackletMemory) -> None:
    r"""
    Tests whether the ``state_dict`` of the memory is correctly initialized.
    """

    for state in STATES_USER + STATES_INTERNAL:
        assert state in memory.states, f"State {state} not found in memory states."


def test_memory_read(memory: ut.TrackletMemory) -> None:
    r"""
    Tests whether the memory resets when the current frame is larger than the stored
    frame.
    """

    assert memory.frame == -1.0, "Memory frame should be 0 at initialization."
    assert memory.count == 0, "Memory count should be 0 at initialization."

    unit_delta = torch.tensor(1.0 / memory.fps)

    # Test reading at frames past the current index 0
    ctx, obs = memory.read(0)

    assert ctx[ut.consts.KEY_FRAME].item() == 0
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], unit_delta)

    ctx, obs = memory.read(1)

    assert ctx[ut.consts.KEY_FRAME].item() == 1
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], 2 * unit_delta)
    ctx, obs = memory.read(5)

    assert ctx[ut.consts.KEY_FRAME].item() == 5
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], 6 * unit_delta)

    with pytest.raises(IndexError, match="negative frame index"):
        memory.read(-1)

    assert memory.frame == -1.0
    assert memory.count == 0

    # Test increasing the frame index and reading at subsequent frames
    memory.frame.fill_(10)
    ctx, obs = memory.read(11)
    assert ctx[ut.consts.KEY_FRAME].item() == 11
    assert torch.allclose(ctx[ut.consts.KEY_DELTA], unit_delta)

    ctx, obs = memory.read(10)
    assert ctx[ut.consts.KEY_FRAME].item() == 10
    assert ctx[ut.consts.KEY_DELTA].item() == 0.0

    assert memory.count == 0
    memory.count.fill_(5)
    assert memory.count == 5
    assert len(memory) == 5
    ctx, obs = memory.read(10)

    assert ctx[ut.consts.KEY_FRAME].item() == 10

    assert ctx[ut.consts.KEY_DELTA].item() == 0.0

    # Test that reading at negative frame indices raises an error when `auto_reset` is `False`
    setattr(memory, "auto_reset", False)

    assert memory.auto_reset is False
    with pytest.raises(IndexError, match="less than current frame"):
        memory.read(9)

    assert memory.frame == 10
    assert memory.count == 5

    # Test that reading at previous frames does not raise an error and resets the counter when `auto_reset` is `True`
    setattr(memory, "auto_reset", True)

    ctx, obs = memory.read(6)

    assert memory.frame == -1, "auto reset was triggered"
    assert memory.count == 0

    assert ctx[ut.consts.KEY_FRAME].item() == 6
    assert ctx[ut.consts.KEY_DELTA].item() == 7 * 1 / memory.fps
