"""
Modules that peform tracking tasks.

A tracker implements an algorithm to find a mapping from detections to 
tracklets, i.e.:

    Tracker: Detections --> Tracklets

Each detection has fields that can be used to assign IDs from Tracklets in the
previous frame to Tracklets in the current frame.

Terminology
-----------
Detections
    All detected structures at the current frame.
Tracklets
    All detections from previous frames, having an unique track ID.
Assignment
    Process that assigns each Detection to a Tracklet.
Lost
    State of a Tracklet that has not been assigned to a detection at the current
    or a previous frame.

"""

__version__ = "4.6.9"

from . import assignment, costs, stages, states  # noqa: F401
from .constants import *  # noqa: F401, F403
from .memory import *  # noqa: F401, F403
from .tracker import *  # noqa: F401, F403
