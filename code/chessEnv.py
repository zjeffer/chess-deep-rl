import config
from re import A
import chess
from chess import Move
import numpy as np

import time
import logging

logging.basicConfig(level=logging.INFO, format=' %(message)s')


class ChessEnv:
    def __init__(self, fen: str = chess.STARTING_FEN):
        """
        Initialize the chess environment
        """
        # the chessboard
        self.fen = fen
        self.reset()

    def reset(self):
        """
        Reset everything
        """
        self.board = chess.Board(self.fen)

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
        return r.astype(bool)

    @staticmethod
    def estimate_winner(board: chess.Board) -> int:
        """
        Estimate the winner of the current node.
        Pawn = 1, Bishop = 3, Rook = 5, Queen = 9
        Positive score = white wins, negative score = black wins
        """
        score = 0
        piece_scores = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        for piece in board.piece_map().values():
            if piece.color == chess.WHITE:
                score += piece_scores[piece.piece_type]
            else:
                score -= piece_scores[piece.piece_type]
        if np.abs(score) > 5:
            if score > 0:
                logging.debug("White wins (estimated)")
                return 0.25
            else:
                logging.debug("Black wins (estimated)")
                return -0.25
        else:
            logging.debug("Draw")
            return 0

    @staticmethod
    def get_piece_amount(board: chess.Board) -> int:
        return len(board.piece_map().values())

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