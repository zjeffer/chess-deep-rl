from enum import Enum


class QueenDirection(Enum):
    # eight directions
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7


class KnightMove(Enum):
    # eight possible knight moves
    NORTH_LEFT = 0
    NORTH_RIGHT = 1
    EAST_UP = 2
    EAST_DOWN = 3
    SOUTH_LEFT = 4
    SOUTH_RIGHT = 5
    WEST_UP = 6
    WEST_DOWN = 7


class Mapping:
    """
    The mapper is a dictionary of tuples.

    * the index is the probability vector's index
    * the first element of the tuple is the direction of the move
    * the second element is how far the piece moves
    """
    mapper = {
        # queen north
        0: (QueenDirection.NORTH, 1),
        1: (QueenDirection.NORTH, 2),
        2: (QueenDirection.NORTH, 3),
        3: (QueenDirection.NORTH, 4),
        4: (QueenDirection.NORTH, 5),
        5: (QueenDirection.NORTH, 6),
        6: (QueenDirection.NORTH, 7),
        # queen northeast
        7: (QueenDirection.NORTHEAST, 1),
        8: (QueenDirection.NORTHEAST, 2),
        9: (QueenDirection.NORTHEAST, 3),
        10: (QueenDirection.NORTHEAST, 4),
        11: (QueenDirection.NORTHEAST, 5),
        12: (QueenDirection.NORTHEAST, 6),
        13: (QueenDirection.NORTHEAST, 7),
        # queen east
        14: (QueenDirection.EAST, 1),
        15: (QueenDirection.EAST, 2),
        16: (QueenDirection.EAST, 3),
        17: (QueenDirection.EAST, 4),
        18: (QueenDirection.EAST, 5),
        19: (QueenDirection.EAST, 6),
        20: (QueenDirection.EAST, 7),
        # queen southeast
        21: (QueenDirection.SOUTHEAST, 1),
        22: (QueenDirection.SOUTHEAST, 2),
        23: (QueenDirection.SOUTHEAST, 3),
        24: (QueenDirection.SOUTHEAST, 4),
        25: (QueenDirection.SOUTHEAST, 5),
        26: (QueenDirection.SOUTHEAST, 6),
        27: (QueenDirection.SOUTHEAST, 7),
        # queen south
        28: (QueenDirection.SOUTH, 1),
        29: (QueenDirection.SOUTH, 2),
        30: (QueenDirection.SOUTH, 3),
        31: (QueenDirection.SOUTH, 4),
        32: (QueenDirection.SOUTH, 5),
        33: (QueenDirection.SOUTH, 6),
        34: (QueenDirection.SOUTH, 7),
        # queen southwest
        35: (QueenDirection.SOUTHWEST, 1),
        36: (QueenDirection.SOUTHWEST, 2),
        37: (QueenDirection.SOUTHWEST, 3),
        38: (QueenDirection.SOUTHWEST, 4),
        39: (QueenDirection.SOUTHWEST, 5),
        40: (QueenDirection.SOUTHWEST, 6),
        41: (QueenDirection.SOUTHWEST, 7),
        # queen west
        42: (QueenDirection.WEST, 1),
        43: (QueenDirection.WEST, 2),
        44: (QueenDirection.WEST, 3),
        45: (QueenDirection.WEST, 4),
        46: (QueenDirection.WEST, 5),
        47: (QueenDirection.WEST, 6),
        48: (QueenDirection.WEST, 7),
        # queen northwest
        49: (QueenDirection.NORTHWEST, 1),
        50: (QueenDirection.NORTHWEST, 2),
        51: (QueenDirection.NORTHWEST, 3),
        52: (QueenDirection.NORTHWEST, 4),
        53: (QueenDirection.NORTHWEST, 5),
        54: (QueenDirection.NORTHWEST, 6),
        55: (QueenDirection.NORTHWEST, 7),
        # knight moves
        56: (KnightMove.NORTH_LEFT, 1),
        57: (KnightMove.NORTH_RIGHT, 1),
        58: (KnightMove.EAST_UP, 1),
        59: (KnightMove.EAST_DOWN, 1),
        60: (KnightMove.SOUTH_RIGHT, 1),
        61: (KnightMove.SOUTH_LEFT, 1),
        62: (KnightMove.WEST_UP, 1),
        63: (KnightMove.WEST_DOWN, 1),
        # underpromotions (knight)
        64: (QueenDirection.NORTHWEST, 1),
        65: (QueenDirection.NORTH, 1),
        66: (QueenDirection.NORTHEAST, 1),
        # underpromotions (bishop)
        67: (QueenDirection.NORTHWEST, 1),
        68: (QueenDirection.NORTH, 1),
        69: (QueenDirection.NORTHEAST, 1),
        # underpromotions (rook)
        70: (QueenDirection.NORTHWEST, 1),
        71: (QueenDirection.NORTH, 1),
        72: (QueenDirection.NORTHEAST, 1),
    }
