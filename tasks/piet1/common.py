from enum import Enum
from typing import Literal

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


# ------------------------------------------------------------------------------
# Colors


bg_color = (0, 0, 0)
start_color = (0, 255, 0)
startDirection_color = (0, 255, 255)
left_color = (255, 0, 0)
right_color = (0, 0, 255)
end_color = (255, 255, 255)


# ------------------------------------------------------------------------------
# Config

max_steps_count = 50
action_types_count = 5
action_steps_count = 30
