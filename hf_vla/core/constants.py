"""
Important constants for VLA training and evaluation.
"""
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]

# Define constants for OmniVLA
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 4
POSE_DIM = 4
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
FUTURE_ACTION_WAYPOINTS = 8
