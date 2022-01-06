from agent import Agent
from chessEnv import ChessEnv
from mcts import MCTS
import time
from tqdm import tqdm

import config
import utils

from test import Test


class Game:
    def __init__(self, env: ChessEnv = ChessEnv(Agent(), Agent())):
        self.env = env
        self.mcts = MCTS(env)

    def play(self):
        pass

    def run_simulations(self, n: int = 1):
        start_time = time.time()
        print(f"Running {n} simulations...")
        # run n simulations
        for _ in tqdm(range(n)):
            self.mcts.run_simulation()
        print("="*40)
        print(f"Amount of simulations: {self.mcts.amount_of_simulations}")
        print(f"Time: {time.time() - start_time}")
        print("="*40)

    def plot_mcts(self):
        self.mcts.plot_tree()


if __name__ == "__main__":

    # # run tests
    # testing = Test()
    # testing.run_tests(n=500)

    white = Agent()
    black = Agent()
    env = ChessEnv(white, black)

    game = Game(env=env)
    # white.model.summary()
    game.run_simulations(n=config.AMOUNT_OF_SIMULATIONS)
    # game.plot_mcts()
