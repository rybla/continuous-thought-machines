from enum import Enum
from typing import Literal

import numpy as np

# ------------------------------------------------------------------------------
# Direction

# type Direction = Literal[0, 1, 2, 3]


class Direction(int, Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# ------------------------------------------------------------------------------
# Action


class Action(int, Enum):
    UP = Direction.UP
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    RIGHT = Direction.RIGHT
    WAIT = 4


action_types_count = 5

# ------------------------------------------------------------------------------
# Colors


bg_color = (0, 0, 0)
start_color = (0, 255, 0)
startDirection_color = (0, 255, 255)
left_color = (255, 0, 0)
right_color = (0, 0, 255)
end_color = (255, 255, 255)

bg_color_np = np.array(bg_color).astype(np.float32) / 255
start_color_np = np.array(start_color).astype(np.float32) / 255
startDirection_color_np = np.array(startDirection_color).astype(np.float32) / 255
left_color_np = np.array(left_color).astype(np.float32) / 255
right_color_np = np.array(right_color).astype(np.float32) / 255
end_color_np = np.array(end_color).astype(np.float32) / 255


# ------------------------------------------------------------------------------
# Config

actions_count = 12
max_segment_length = 4
grid_size = 8