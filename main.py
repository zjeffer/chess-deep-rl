from agent import Agent
from chessEnv import ChessEnv
from mcts import MCTS

class Game:
	def __init__(self, env=ChessEnv()):
		self.env = env
		self.white = Agent()
		self.black = Agent()
		self.mcts = MCTS()

	def play(self):
		pass

	def run_simulations(self, n:int = 1):
		# run n simulations
		for i in range(n):
			self.mcts.run_simulation()
		print("="*40)
		print(f"Amount of simulations: {self.mcts.amount_of_simulations}")
		print(f"Amount of expansions: {self.mcts.amount_of_expansions}")
		print("="*40)

if __name__ == "__main__":
	env = ChessEnv()
	game = Game(env=env)
	game.run_simulations(n=1)
