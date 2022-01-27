from agent import Agent
from chessEnv import ChessEnv
from game import Game

class Evaluation:
	def __init__(self, model_1_path: str, model_2_path: str):
		self.model_1 = model_1_path
		self.model_2 = model_2_path


	def evaluate(self, n: int):
		"""
		For n games, let the two models play each other and keep a score
		"""
		score = {
			"model_1": 0,
			"model_2": 0,
			"amount_of_draws": 0
		}
		agent_1 = Agent(local_predictions=True, model_path=self.model_1)
		agent_2 = Agent(local_predictions=True, model_path=self.model_2)
		for i in range(n):
			print(f"{'*'*10}\nPlaying match {i+1}/{n}\n{'*'*10}")
			game = Game(ChessEnv(), agent_1, agent_2)
			# play deterministally
			result = game.play_one_game(stochastic=False)
			if result == 0: score["amount_of_draws"] += 1
			elif result == 1: score["model_1"] += 1
			else: score["model_2"] += 1
			# turn around the colors
			game = Game(ChessEnv(), agent_2, agent_1)
			result = game.play_one_game(stochastic=False)
			if result == 0: score["amount_of_draws"] += 1
			elif result == 1: score["model_2"] += 1
			else: score["model_1"] += 1
		
		return f"Evaluated these models: Model 1 = {self.model_1}, Model 2 = {self.model_2}\n" + \
		f"The results: \nModel 1: {score['model_1']} \nModel 2: {score['model_2']} \nDraws: {score['amount_of_draws']}"


if __name__ == "__main__":
	# get args
	import argparse
	parser = argparse.ArgumentParser(description="Evaluate two models")
	parser.add_argument("model_1", help="Path to model 1", type=str)
	parser.add_argument("model_2", help="Path to model 2", type=str)
	parser.add_argument("nr_games", help="Number of games to play (x2: every model plays both white and black)", type=int)
	args = parser.parse_args()

	# args to dict
	args = vars(args)
	
	evaluation = Evaluation(args["model_1"], args["model_2"])
	print(evaluation.evaluate(int(args["nr_games"])))