import config
from re import A
import chess
from chess import Move
import numpy as np

import time
import logging
logging.basicConfig(level=logging.INFO)


class ChessEnv:
    def __init__(self):
        """
        Initialize the chess environment
        """
        # the chessboard
        self.board = chess.Board()

    def reset(self):
        """
        Reset everything
        """
        self.board = chess.Board()

    @staticmethod
    def state_to_input(fen: str) -> np.ndarray(config.INPUT_SHAPE):
        """
        Convert board to a state that is interpretable by the model
        """

        board = chess.Board(fen)

        # 1. is it white's turn? (1x8x8)
        is_white_turn = np.ones((8, 8)) if board.turn else np.zeros((8, 8))

        # 2. castling rights (4x8x8)
        castling = np.asarray([
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
        ])

        # 3. repitition counter
        counter = np.ones(
            (8, 8)) if board.can_claim_fifty_moves() else np.zeros((8, 8))

        # create new np array
        arrays = []
        for color in chess.COLORS:
            # 4. player 1's pieces (6x8x8)
            # 5. player 2's pieces (6x8x8)
            for piece_type in chess.PIECE_TYPES:
                # 6 arrays of 8x8 booleans
                array = np.zeros((8, 8))
                for index in list(board.pieces(piece_type, color)):
                    # row calculation: 7 - index/8 because we want to count from bottom left, not top left
                    array[7 - int(index/8)][index % 8] = True
                arrays.append(array)
        arrays = np.asarray(arrays)

        # 6. en passant square (8x8)
        en_passant = np.zeros((8, 8))
        if board.has_legal_en_passant():
            en_passant[7 - int(board.ep_square/8)][board.ep_square % 8] = True

        r = np.array([is_white_turn, *castling,
                     counter, *arrays, en_passant]).reshape((1, *config.INPUT_SHAPE))
        # memory management
        del board
        return r

    def __str__(self):
        """
        Print the board
        """
        return str(chess.Board(self.board))

    def step(self, action: Move) -> chess.Board:
        """
        Perform a step in the game
        """
        self.board.push(action)
        return self.board