from agent import Agent
from chessEnv import ChessEnv
from mcts import MCTS
from tqdm import tqdm
import config


class Game:
    def __init__(self, env: ChessEnv, white: Agent, black: Agent):
        self.env = env
        self.white = white
        self.black = black

        self.turn = True # True = white, False = black


    def play_one_move(self):
        if self.turn:
            self.white.mcts = MCTS(self.white)
            self.white.run_simulations(n = config.SIMULATIONS_PER_MOVE)
            # TODO play best move from simulation

        else:
            self.black.run_simulations(n = config.SIMULATIONS_PER_MOVE)
            # TODO: same for black
        self.turn = not self.turn


if __name__ == "__main__":
    white = Agent()
    black = Agent()
    env = ChessEnv()

    game = Game(env=env, white=white, black=black)
    game.play_one_move()
