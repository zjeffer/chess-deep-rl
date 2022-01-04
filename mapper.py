from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from chess import PieceType
import numpy as np


class QueenDirection(Enum):
    # eight directions
    NORTHWEST = 0
    NORTH = 1
    NORTHEAST = 2
    EAST = 3
    SOUTHEAST = 4
    SOUTH = 5
    SOUTHWEST = 6
    WEST = 7


class KnightMove(Enum):
    # eight possible knight moves
    NORTH_LEFT = 0  # diff == -15
    NORTH_RIGHT = 1  # diff == -17
    EAST_UP = 2  # diff == -6
    EAST_DOWN = 3  # diff == 10
    SOUTH_RIGHT = 4  # diff == 15
    SOUTH_LEFT = 5  # diff == 17
    WEST_DOWN = 6  # diff == 6
    WEST_UP = 7  # diff == -10


class UnderPromotion(Enum):
    KNIGHT = 0
    BISHOP = 1
    ROOK = 2


class Mapping:
    """
    The mapper is a dictionary of moves.

    * the index is the type of move
    * the value is the plane's index, or an array of plane indices (for distance)
    """
    # knight moves from north_left to west_up (clockwise)
    knight_mappings = [-15, -17, -6, 10, 15, 17, 6, -10]

    def get_index(self, piece_type: PieceType, direction: Enum, distance: int = 1) -> int:
        if piece_type == PieceType.KNIGHT:
            return 56 + KnightMove(direction).value
        else:
            return QueenDirection(direction) * 8 + distance

    @staticmethod
    def get_underpromotion_move(piece_type: PieceType, from_square: int, to_square: int) -> Tuple[UnderPromotion, int]:
        piece_type = UnderPromotion(piece_type - 2)
        diff = from_square - to_square
        if to_square < 8:
            # black promotes (1st rank)
            direction = diff - 8
        elif to_square > 55:
            # white promotes (8th rank)
            direction = diff + 8
        return (piece_type, direction)

    @staticmethod
    def get_knight_move(from_square: int, to_square: int) -> KnightMove:
        return KnightMove(Mapping.knight_mappings.index(from_square - to_square))

    @staticmethod
    def get_queenlike_move(from_square: int, to_square: int) -> Tuple[QueenDirection, int]:
        diff = from_square - to_square
        if diff % 8 == 0:
            # north and south
            if diff > 0:
                direction = QueenDirection.SOUTH
            else:
                direction = QueenDirection.NORTH
            distance = int(diff / 8)
        elif diff % 9 == 0:
            # southwest and northeast
            if diff > 0:
                direction = QueenDirection.SOUTHWEST
            else:
                direction = QueenDirection.NORTHEAST
            distance = np.abs(int(diff / 8))
        elif from_square // 8 == to_square // 8:
            # east and west
            if diff > 0:
                direction = QueenDirection.WEST
            else:
                direction = QueenDirection.EAST
            distance = np.abs(diff)
        elif diff % 7 == 0:
            if diff > 0:
                direction = QueenDirection.SOUTHEAST
            else:
                direction = QueenDirection.NORTHWEST
            distance = np.abs(int(diff / 8)) + 1
        else:
            raise Exception("Invalid queen-like move")
        return (direction, distance)

    mapper = {
        # queens
        QueenDirection.NORTHWEST: [0, 1, 2, 3, 4, 5, 6],
        QueenDirection.NORTH: [7, 8, 9, 10, 11, 12, 13],
        QueenDirection.NORTHEAST: [14, 15, 16, 17, 18, 19, 20],
        QueenDirection.EAST: [21, 22, 23, 24, 25, 26, 27],
        QueenDirection.SOUTHEAST: [28, 29, 30, 31, 32, 33, 34],
        QueenDirection.SOUTH: [35, 36, 37, 38, 39, 40, 41],
        QueenDirection.SOUTHWEST: [42, 43, 44, 45, 46, 47, 48],
        QueenDirection.WEST: [49, 50, 51, 52, 53, 54, 55],
        # knights
        KnightMove.NORTH_LEFT: 56,
        KnightMove.NORTH_RIGHT: 57,
        KnightMove.EAST_UP: 58,
        KnightMove.EAST_DOWN: 59,
        KnightMove.SOUTH_RIGHT: 60,
        KnightMove.SOUTH_LEFT: 61,
        KnightMove.WEST_DOWN: 62,
        KnightMove.WEST_UP: 63,
        # underpromotions
        UnderPromotion.KNIGHT: [64, 65, 66],
        UnderPromotion.BISHOP: [67, 68, 69],
        UnderPromotion.ROOK: [70, 71, 72]
    }

    # mapper = {
    #     # TODO: fix mapping: northwest should be first
    #     # queen northwest
    #     49: (QueenDirection.NORTHWEST, 1),
    #     50: (QueenDirection.NORTHWEST, 2),
    #     51: (QueenDirection.NORTHWEST, 3),
    #     52: (QueenDirection.NORTHWEST, 4),
    #     53: (QueenDirection.NORTHWEST, 5),
    #     54: (QueenDirection.NORTHWEST, 6),
    #     55: (QueenDirection.NORTHWEST, 7),
    #     # TODO: is there a better way to do this?
    #     # queen north
    #     0: (QueenDirection.NORTH, 1),
    #     1: (QueenDirection.NORTH, 2),
    #     2: (QueenDirection.NORTH, 3),
    #     3: (QueenDirection.NORTH, 4),
    #     4: (QueenDirection.NORTH, 5),
    #     5: (QueenDirection.NORTH, 6),
    #     6: (QueenDirection.NORTH, 7),
    #     # queen northeast
    #     7: (QueenDirection.NORTHEAST, 1),
    #     8: (QueenDirection.NORTHEAST, 2),
    #     9: (QueenDirection.NORTHEAST, 3),
    #     10: (QueenDirection.NORTHEAST, 4),
    #     11: (QueenDirection.NORTHEAST, 5),
    #     12: (QueenDirection.NORTHEAST, 6),
    #     13: (QueenDirection.NORTHEAST, 7),
    #     # queen east
    #     14: (QueenDirection.EAST, 1),
    #     15: (QueenDirection.EAST, 2),
    #     16: (QueenDirection.EAST, 3),
    #     17: (QueenDirection.EAST, 4),
    #     18: (QueenDirection.EAST, 5),
    #     19: (QueenDirection.EAST, 6),
    #     20: (QueenDirection.EAST, 7),
    #     # queen southeast
    #     21: (QueenDirection.SOUTHEAST, 1),
    #     22: (QueenDirection.SOUTHEAST, 2),
    #     23: (QueenDirection.SOUTHEAST, 3),
    #     24: (QueenDirection.SOUTHEAST, 4),
    #     25: (QueenDirection.SOUTHEAST, 5),
    #     26: (QueenDirection.SOUTHEAST, 6),
    #     27: (QueenDirection.SOUTHEAST, 7),
    #     # queen south
    #     28: (QueenDirection.SOUTH, 1),
    #     29: (QueenDirection.SOUTH, 2),
    #     30: (QueenDirection.SOUTH, 3),
    #     31: (QueenDirection.SOUTH, 4),
    #     32: (QueenDirection.SOUTH, 5),
    #     33: (QueenDirection.SOUTH, 6),
    #     34: (QueenDirection.SOUTH, 7),
    #     # queen southwest
    #     35: (QueenDirection.SOUTHWEST, 1),
    #     36: (QueenDirection.SOUTHWEST, 2),
    #     37: (QueenDirection.SOUTHWEST, 3),
    #     38: (QueenDirection.SOUTHWEST, 4),
    #     39: (QueenDirection.SOUTHWEST, 5),
    #     40: (QueenDirection.SOUTHWEST, 6),
    #     41: (QueenDirection.SOUTHWEST, 7),
    #     # queen west
    #     42: (QueenDirection.WEST, 1),
    #     43: (QueenDirection.WEST, 2),
    #     44: (QueenDirection.WEST, 3),
    #     45: (QueenDirection.WEST, 4),
    #     46: (QueenDirection.WEST, 5),
    #     47: (QueenDirection.WEST, 6),
    #     48: (QueenDirection.WEST, 7),
    #     # knight moves
    #     56: (KnightMove.NORTH_LEFT, 1),
    #     57: (KnightMove.NORTH_RIGHT, 1),
    #     58: (KnightMove.EAST_UP, 1),
    #     59: (KnightMove.EAST_DOWN, 1),
    #     60: (KnightMove.SOUTH_RIGHT, 1),
    #     61: (KnightMove.SOUTH_LEFT, 1),
    #     62: (KnightMove.WEST_UP, 1),
    #     63: (KnightMove.WEST_DOWN, 1),
    #     # underpromotions (knight)
    #     64: (QueenDirection.NORTHWEST, 1),
    #     65: (QueenDirection.NORTH, 1),
    #     66: (QueenDirection.NORTHEAST, 1),
    #     # underpromotions (bishop)
    #     67: (QueenDirection.NORTHWEST, 1),
    #     68: (QueenDirection.NORTH, 1),
    #     69: (QueenDirection.NORTHEAST, 1),
    #     # underpromotions (rook)
    #     70: (QueenDirection.NORTHWEST, 1),
    #     71: (QueenDirection.NORTH, 1),
    #     72: (QueenDirection.NORTHEAST, 1),
    # }
