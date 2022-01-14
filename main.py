from email import header
from re import search
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
import pandas as pd

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

    @staticmethod
    def get_winner(result: str) -> int:
        return 1 if result == "1-0" else - 1 if result == "0-1" else 0


    @utils.timer_function
    def play_one_game(self, stochastic: bool = True) -> int:
        # reset everything
        self.reset()
        # add a new memory entry
        self.memory.append([])
        # show the board
        print(self.env.board)
        # counter to check amount of moves played. if above limit, estimate winner
        counter, full_game = 0, True
        while not self.env.board.is_game_over():
            self.play_move(stochastic=stochastic)
            counter += 1
            if counter > config.MAX_GAME_MOVES:
                # estimate the winner based on piece values
                winner = ChessEnv.estimate_winner(self.env.board)
                print(f"Game over by move limit ({config.MAX_GAME_MOVES}). Result: {winner}")
                full_game = False
                break
        if full_game:
            # get the winner based on the result of the game
            winner = Game.get_winner(self.env.board.result())
            print(f"Game over. Result: {winner}")
        # save game result to memory for all games
        for index, element in enumerate(self.memory[-1]):
            self.memory[-1][index] = (element[0], element[1], winner)

        game = ChessGame()
        # set starting position
        game.setup(self.env.fen)
        # add moves
        node = game.add_variation(self.env.board.move_stack[0])
        for move in self.env.board.move_stack[1:]:
            print(move)
            node = node.add_variation(move)
        # print pgn
        print(game)

        # save memory to file
        self.save_game(name="game")

        return winner

    def play_move(self, stochastic: bool = True) -> None:
        # whose turn is it
        current_player = self.white if self.turn else self.black

        # create tree with root node == current board
        current_player.mcts = MCTS(
            current_player, state=self.env.board.fen())
        # play n simulations from the root node
        current_player.run_simulations(n=config.SIMULATIONS_PER_MOVE)

        # print(f"Amount of children in tree: {len(current_player.mcts.root.get_all_children())}")
        moves = current_player.mcts.root.edges

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
        # create dictionary of moves and their probabilities
        search_probabilities = {
            e.action.uci(): e.N / sum_move_visits for e in moves}
        # winner gets added after game is over
        self.memory[-1].append((state, search_probabilities, None))

    def save_game(self, name: str = "game") -> None:
        # the game id consist of game + datetime
        game_id = f"{name}-{str(uuid.uuid4())[:8]}"
        np.save(os.path.join(config.MEMORY_DIR, game_id), self.memory[-1])
        print(
            f"Game saved to {os.path.join(config.MEMORY_DIR, game_id)}.npy")
        print(f"Memory size: {len(self.memory)}")


    @utils.timer_function
    def train_puzzles(self, puzzles: pd.DataFrame):
        """
        Create positions from puzzles (fen strings) and let the MCTS figure out how to solve them.
        The saved positions can be used to train the neural network.
        """
        print(f"Training on {len(puzzles)} puzzles")
        for puzzle in puzzles.itertuples():
            self.env.fen = puzzle.fen
            self.env.reset()
            # play the first move
            moves = puzzle.moves.split(" ")
            self.env.board.push_uci(moves.pop(0))
            print(f"Puzzle to solve ({puzzle.rating} ELO): {self.env.fen}")
            print(self.env.board)
            print(f"Correct solution: {moves} ({len(moves)} moves)")
            self.memory.append([])
            counter, solved = 0, True
            while not self.env.board.is_game_over():
                # deterministically choose the next move (we want no exploration here)
                self.play_move(stochastic=False)
                counter += 1
                if counter > config.MAX_PUZZLE_MOVES:
                    print("Puzzle could not be solved within the move limit")
                    solved = False
                    break
            if not solved: 
                continue
            print(f"Puzzle complete. Ended after {counter} moves: {self.env.board.result()}")
            # save game result to memory for all games
            winner = Game.get_winner(self.env.board.result())
            for index, element in enumerate(self.memory[-1]):
                self.memory[-1][index] = (element[0], element[1], winner)

            game = ChessGame()
            # set starting position
            game.setup(self.env.fen)
            # add moves
            node = game.add_variation(self.env.board.move_stack[0])
            for move in self.env.board.move_stack[1:]:
                print(move)
                node = node.add_variation(move)
            # print pgn
            print(game)

            # save memory to file
            self.save_game(name="puzzle")

    def create_puzzle_set(self, filename: str):
        puzzles = pd.read_csv(filename, header=None)
        # shuffle pandas rows
        puzzles = puzzles.sample(frac=1).reset_index(drop=True)
        # drop unnecessary columns
        puzzles = puzzles.drop(columns=[0, 4, 5, 6, 8])
        # set column names
        puzzles.columns = ["fen", "moves", "rating", "type"]
        # only keep puzzles where type contains "mate"
        puzzles = puzzles[puzzles["type"].str.contains("mateIn2")]
        game.train_puzzles(puzzles)

    def create_training_set(self):
        counter = {"white": 0, "black": 0, "draw": 0}
        while True:
            winner = game.play_one_game(stochastic=True)
            if winner == 1:
                counter["white"] += 1
            elif winner == -1:
                counter["black"] += 1
            else:
                counter["draw"] += 1
            print(
                f"Game results: {counter['white']} - {counter['black']} - {counter['draw']}")


if __name__ == "__main__":
    model_path = os.path.join(config.MODEL_FOLDER, "model.h5")
    white = Agent(model_path)
    black = Agent(model_path)

    # test with a mate in 1 game (black to play)
    # env = ChessEnv("5K2/r1r5/p2p4/k1pP4/2P5/8/8/8 b - - 1 2")

    env = ChessEnv()
    game = Game(env=env, white=white, black=black)
    game.create_training_set()
    # game.create_puzzle_set(filename="puzzles/lichess_db_puzzle.csv")
    
