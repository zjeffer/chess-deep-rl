from keras.backend import reverse
from agent import Agent
from chessEnv import ChessEnv
from mcts import MCTS
from tqdm import tqdm
import config
import chess

class Game:
    def __init__(self, env: ChessEnv, white: Agent, black: Agent):
        self.env = env
        self.white = white
        self.black = black

        self.turn = True # True = white, False = black


    def play_one_move(self):
        # whose turn is it
        current_player = self.white if self.turn else self.black

        # create tree with root node == current board
        current_player.mcts = MCTS(current_player, state=self.env.board.fen())
        # play n simulations from the root node
        current_player.run_simulations(n = config.SIMULATIONS_PER_MOVE)
        # play best move from simulations
        # TODO: try stochastic policy instead of deterministic
        best_moves = sorted(current_player.mcts.root.edges, key=lambda e: e.N, reverse=True)

        # print("5 Best moves:")
        # best_moves = best_moves[:5]
        # best_moves.reverse()
        # for i, move in enumerate(best_moves):
        #     print(f"{i+1}: {move.action}")

        # print(f"Amount of children: {len(current_player.mcts.root.get_all_children())}")

        # make the move
        best_move = best_moves[0]
        new_board = self.env.step(best_move.action)
        print(f"{'White' if self.turn else 'Black'} played {best_move.action}")
        print(new_board)
        print(f"Value according to white: {self.white.mcts.root.value}")
        print(f"Value according to black: {self.black.mcts.root.value}")
        # TODO: save each move to memory

        # switch turn
        self.turn = not self.turn


if __name__ == "__main__":
    white = Agent()
    black = Agent()
    env = ChessEnv()

    game = Game(env=env, white=white, black=black)
    for _ in range(10):
        game.play_one_move()
    # print("Plotting tree")
    # game.white.mcts.plot_tree()
