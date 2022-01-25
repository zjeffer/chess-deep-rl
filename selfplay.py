import logging
from multiprocessing import Pool
import time
from agent import Agent
from chessEnv import ChessEnv
from game import Game
import config
import numpy as np
import os
import chess

# set logging config
logging.basicConfig(level=logging.INFO, format=' %(message)s')

def setup(starting_position: str = chess.STARTING_FEN) -> Game:
    """
    Setup function to set up a game. 
    This can be used in both the self-play and puzzle solving function
    """
    # set different random seeds for each process
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # create environment and game
    env = ChessEnv(fen=starting_position)

    # create agents
    # model_path = os.path.join(config.MODEL_FOLDER, "model.h5")
    white = Agent(state=env.board.fen())
    black = Agent(state=env.board.fen())

    
    return Game(env=env, white=white, black=black)

def multiprocessed_self_play(_ = None):
    """
    Continuously play games against itself
    """
    game = setup()

    # play games continuously
    while True:
        game.play_one_game(stochastic=True)
    # game.create_puzzle_set(filename="puzzles/lichess_db_puzzle.csv")

def multiprocessed_puzzle_solver(puzzles):
    """
    Continuously solve puzzles 
    """
    game = setup()

    # solve puzzles continuously
    while True:
        # shuffle pandas rows
        puzzles = puzzles.sample(frac=1).reset_index(drop=True)
        game.train_puzzles()

if __name__ == "__main__":
    # the amount of games to play simultaneously
    p_count = 4


    with Pool(processes=p_count) as pool:
        pool.map(multiprocessed_self_play, [None for _ in range(p_count)])

    # multiprocessed puzzle solver
    # with Pool(processes=p_count) as pool:
    #     puzzles = Game.create_puzzle_set(filename="puzzles/lichess_db_puzzle.csv", type="mateIn1")
    #     pool.map(multiprocessed_puzzle_solver, [puzzles for _ in range(p_count)])
    
    
