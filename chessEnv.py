import chess
import numpy as np
from agent import Agent

# class for chess environment


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
	def board_to_booleans(board: chess.Board):
		"""
		Convert the board to a boolean representation
		"""
		booleans = []
		# for every player
		for color in chess.COLORS:
			# create an array of boards
			booleans.append([])
			for piece_type in chess.PIECE_TYPES:
				indexes = list(board.pieces(piece_type, color))
				# create the array for the current piece type
				array = [[0 for x in range(8)] for y in range(8)]
				# for every index, set the array to true
				for index in indexes:
					array[int(index/8)][index % 8] = 1
				booleans[-1].append(array)
		return booleans

	@staticmethod
	def board_to_state(board: chess.Board):
		"""
		Convert board to a state that is interpretable by the model
		"""

		# put history of chess board into a list
		boards = []
		curr_board = board.copy()
		# put last 10 moves in state
		history_amount = 10
		# if there are less than 10 moves, put all of them in state
		if len(curr_board.move_stack) < history_amount:
			history_amount = len(curr_board.move_stack)
		for move in board.move_stack[-history_amount:][::-1]:
			curr_board.undo(move)
			booleans = ChessEnv.board_to_booleans(curr_board)
			boards.append(booleans)
		# TODO: Right now, the boards[] array contains the last 10 boards,
		# but do i need to separate the colors like AlphaGo Zero does?

		return boards

	@staticmethod
	def print_board(board: list):
		"""
		Print the board
		"""
		for row in board:
			for col in row:
				print(col)

	def move(self, action: chess.Move):
		""" 
		Perform an action on the board
		"""
		pass


if __name__ == "__main__":
	chessEnv = ChessEnv()
	state = chessEnv.board_to_state(chessEnv.board)
	ChessEnv.print_board(state[0])
	print(np.array(state).shape)
