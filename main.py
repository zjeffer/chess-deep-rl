from agent import Agent
from chessEnv import ChessEnv
from mcts import MCTS
import time
from tqdm import tqdm

import config

# graphing mcts
import graphviz
from graphviz import Digraph

class Game:
	def __init__(self, env: ChessEnv=ChessEnv()):
		self.env = env
		self.mcts = MCTS(env)

	def play(self):
		pass

	def run_simulations(self, n:int = 1):
		start_time = time.time()
		print(f"Running {n} simulations...")
		# run n simulations
		for _ in tqdm(range(n)):
			self.mcts.run_simulation()
		print("="*40)
		print(f"Amount of simulations: {self.mcts.amount_of_simulations}")
		print(f"Amount of expansions: {self.mcts.amount_of_expansions}")
		print(f"Time: {time.time() - start_time}")
		print("="*40)


	def plot_mcts(self):
		# tree plotting
		dot = Digraph(comment='Chess MCTS Tree')
		print(f"Amount of nodes in tree: {len(self.mcts.root.get_all_children())}")
		print(f"Plotting tree...")
		for node in tqdm(self.mcts.root.get_all_children()):
			dot.node(str(node.state.fen()), label="*")
			for child in node.children:
				dot.edge(str(node.state.fen()), str(child.state.fen()), label=str(child.action))
		dot.save('mcts_tree.gv')


if __name__ == "__main__":
	white = Agent()
	black = Agent()
	env = ChessEnv(white, black)

	game = Game(env=env)
	game.run_simulations(n=config.AMOUNT_OF_SIMULATIONS)
	# game.plot_mcts()
