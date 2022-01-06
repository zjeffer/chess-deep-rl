from re import A
import chess
import numpy as np
from agent import Agent

import time
import logging
logging.basicConfig(level=logging.INFO)
import config


class ChessEnv:
    def __init__(self, white: Agent, black: Agent):
        """
        Initialize the chess environment
        """
        # the chessboard
        self.board = chess.Board()
        self.white = white
        self.black = black

    def reset(self):
        """
        Reset everything
        """
        self.board = chess.Board()

    @staticmethod
    def state_to_input(board: chess.Board) -> np.ndarray(config.INPUT_SHAPE):
        """
        Convert board to a state that is interpretable by the model
        """

        # TODO: optimize?
        start_time = time.time()

        # 1. is it white's turn? (1x8x8)
        is_white_turn = np.array([board.turn for _ in range(64)]).reshape(8, 8)

        # 2. is it black's turn? (1x8x8)
        # opposite of is_white_turn
        is_black_turn = np.invert(is_white_turn)

        logging.debug(
            f"*** Turn: {(time.time() - start_time):.6f} seconds ***")
        start_time = time.time()

        # 2. castling rights (4x8x8)
        castling = np.array(
            [np.array([board.has_queenside_castling_rights(chess.WHITE) for _ in range(64)]).reshape(8, 8),
             np.array([board.has_kingside_castling_rights(chess.WHITE)
                      for _ in range(64)]).reshape(8, 8),
             np.array([board.has_queenside_castling_rights(chess.BLACK)
                      for _ in range(64)]).reshape(8, 8),
             np.array([board.has_kingside_castling_rights(chess.BLACK) for _ in range(64)]).reshape(8, 8)])

        logging.debug(
            f"*** Castling: {(time.time() - start_time):.6f} seconds ***")
        start_time = time.time()

        # 3. repitition counter
        is_repitition = board.is_repetition()
        counter = np.array([is_repitition for _ in range(64)]).reshape(8, 8)

        logging.debug(
            f"*** Repitition counter: {(time.time() - start_time):.6f} seconds ***")
        start_time = time.time()

        arrays = []
        for color in chess.COLORS:
            # 4. player 1's pieces (6x8x8)
            # 5. player 2's pieces (6x8x8)
            for piece_type in chess.PIECE_TYPES:
                # 6 arrays of 8x8 booleans
                array = np.array([False for _ in range(64)]).reshape(8, 8)
                for index in list(board.pieces(piece_type, color)):
                    # row calculation: 7 - index/8 because we want to count from bottom left, not top left
                    array[7 - int(index/8)][index % 8] = True
                arrays.append(array)
        arrays = np.asarray(arrays)

        logging.debug(
            f"*** Pieces for both players: {(time.time() - start_time):.6f} seconds ***")
        start_time = time.time()

        # 6. en passant square (8x8)
        en_passant = np.array([False for _ in range(64)]).reshape(8, 8)
        if board.has_legal_en_passant():
            en_passant[7 - int(board.ep_square/8)][board.ep_square % 8] = True

        logging.debug(
            f"*** En passant: {(time.time() - start_time):.6f} seconds ***")

        r = np.array([is_white_turn, is_black_turn, *castling, counter, *arrays, en_passant])
        r = r.reshape(1, 8, 8, 20)
        return r

    def __str__(self):
        """
        Print the board
        """
        return str(self.board)

    def move(self, action: chess.Move):
        """ 
        Perform an action on the board
        """
        pass
