from rlmodelbuilder import RLModelBuilder
from chessEnv import ChessEnv
from mcts import MCTS


class Agent:
	def __init__(self):
		# 2 players, 6 pieces, 8x8 board
		n = 8  # board size
		t = 8  # amount of timesteps
		m = 2 * 6 + 1  # pieces for every player + the square for en passant
		# additional values: move counter, repitition counter, side to move, castling rights for every side and every player
		l = 1 + 1 + 1 + (2*2)
		input_shape = (n, n, m*t+l)

		# the model has 2 outputs: policy and value
		# ouput_shape[0] should be the number of possible moves (
		#       * 8x8 board: 8*8=64 possible actions
		#       * 56 possible queen-like moves (horizontal/vertical/diagonal)
		#       * 8 possible knight moves (every direction)
		#       * 9 possible underpromotions
		# ouput_shape[1] should be 1: a scalar value (v)
		output_shape = (8*8*(56+8+9), 1)

		# create the model (AlphaZero used 19 residual blocks)
		# TODO: change the number of residual blocks if necessary
		model_builder = RLModelBuilder(
			input_shape, output_shape, nr_hidden_layers=19)
		self.model = model_builder.build_model()

		# mcts tree
		self.mcts = MCTS()

		# memory
		self.memory = []

	def reset(self):
		self.chess_env.reset()

	def play_one_move(self):
		pass

	def save_to_memory(self, game):
		if len(self.memory) >= self.max_replay_memory:
			self.memory.pop(0)
		self.memory.append(game)

	def evaluate_network(self, best_model, amount=400):
		"""
		Test to see if new network is stronger than the current best model
		Do this by playing x games. If the new network wins more, it is the new best model 
		"""
		pass


if __name__ == "__main__":
	env = ChessEnv()
	agent = Agent()
	# opponent = Agent()

	print(agent.model.summary())

	# play a game
