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
	evaluation = Evaluation("models/model_4.h5", "models/model_5.h5")
	print(evaluation.evaluate(5))