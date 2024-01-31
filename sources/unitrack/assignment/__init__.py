"""
This package implements modules that solve a Linear Assignment Problem (LAP), 
where the minimum cost must be computed over a cost-matrix.
"""

from __future__ import annotations

# from .hungarian import *
# from .auction import *
from .base_assignment import *
from .greedy import *
from .jonker import *
