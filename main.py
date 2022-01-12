import uuid
from agent import Agent
from chessEnv import ChessEnv
from mcts import MCTS
import config
import chess
from chess.pgn import Game as ChessGame
import numpy as np
import os
import utils


class Game:
    def __init__(self, env: ChessEnv, white: Agent, black: Agent):
        self.env = env
        self.white = white
        self.black = black

        self.memory = []

        self.reset()

    def reset(self):
        self.env.reset()
        self.turn = self.env.board.turn  # True = white, False = black

    def play_one_game(self, stochastic: bool = True) -> None:
        self.reset()
        self.memory.append([])
        while not self.env.board.is_game_over():
            self.play_moves(stochastic=stochastic)
        print(f"Game over. Result: {self.env.board.result()}")
        # save game result to memory for all games
        # Returns ``1-0``, ``0-1`` or ``1/2-1/2``.
        winner = 1 if self.env.board.result() == "1-0" else - \
            1 if self.env.board.result() == "0-1" else 0
        for index, element in enumerate(self.memory[-1]):
            self.memory[-1][index] = (element[0], element[1], winner)

        game = ChessGame()
        # starting position
        game.setup(self.env.fen)
        # add moves
        node = game.add_variation(self.env.board.move_stack[0])
        for move in self.env.board.move_stack[1:]:
            print(move)
            node = node.add_variation(move)
        # print pgn
        print(game)

        # save memory to file
        # self.save_game()

    def play_moves(self, n: int = 1, stochastic: bool = True) -> None:
        for _ in range(n):
            # whose turn is it
            current_player = self.white if self.turn else self.black

            # create tree with root node == current board
            current_player.mcts = MCTS(
                current_player, state=self.env.board.fen())
            # play n simulations from the root node
            current_player.run_simulations(n=config.SIMULATIONS_PER_MOVE)
            # play best move from simulations

            print(
                f"Amount of children in tree: {len(current_player.mcts.root.get_all_children())}")
            moves = current_player.mcts.root.edges
            for move in moves:
                print(f"#### MOVE: {move}")

            # TODO: save each move to memory
            # TODO: check if storing input state is faster/less space-consuming than storing the fen string
            self.save_to_memory(self.env.board.fen(), moves)

            sum_move_visits = sum(e.N for e in moves)
            probs = [e.N / sum_move_visits for e in moves]
            if stochastic:
                # stochastically choose the best move
                best_move = np.random.choice(moves, p=probs)
            else:
                # choose the move based on the highest visit count
                best_move = moves[np.argmax(probs)]

            # play the move
            print(
                f"{'White' if self.turn else 'Black'} played  {self.env.board.fullmove_number}. {best_move.action}")
            new_board = self.env.step(best_move.action)
            print(new_board)
            print(f"Value according to white: {self.white.mcts.root.value}")
            print(f"Value according to black: {self.black.mcts.root.value}")

            # switch turn
            self.turn = not self.turn

    def save_to_memory(self, state, moves) -> None:
        sum_move_visits = sum(e.N for e in moves)
        search_probabilities = [
            {e.action.uci(): e.N / sum_move_visits} for e in moves]
        # winner gets added after game is over
        self.memory[-1].append((state, search_probabilities, None))

    @utils.timer_function
    def save_game(self):
        # the game id consist of game + datetime
        game_id = f"game-{uuid.uuid4()[:8]}"
        np.save(os.path.join(config.MEMORY_DIR, game_id), self.memory)
        print(
            f"Game saved to {os.path.join(config.MEMORY_DIR, game_id)}.npy")


if __name__ == "__main__":
    white = Agent()
    black = Agent()
    # env = ChessEnv("8/8/8/8/k7/r7/p7/K7 w - - 0 1")

    # test with a mate in 1 game (black to play)
    # env = ChessEnv("5K2/r1r5/p2p4/k1pP4/2P5/8/8/8 b - - 1 2")
    env = ChessEnv("8/p7/8/R1rR2p1/4pk1p/7P/P4K2/8 w - - 1 35")

    game = Game(env=env, white=white, black=black)
    # game.play_moves(10)
    game.play_one_game(stochastic=False)
    # print("Plotting tree")
    # game.white.mcts.plot_tree()
