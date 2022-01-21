import random
import threading
import numpy as np
from chessEnv import ChessEnv
from game import Game
from agent import Agent
import logging
logging.basicConfig(level=logging.INFO, format=" %(message)s")
logging.disable(logging.WARN)

from GUI.display import GUI

class Main:
    def __init__(self, player: bool = np.random.choice([True, False])):
        self.player = player
        
        # create an agent for the opponent
        self.opponent = Agent(local_predictions=True, model_path="models/model_all_data.h5")

        if self.player:
            self.game = Game(ChessEnv(), None, self.opponent)
        else:
            self.game = Game(ChessEnv(), self.opponent, None)

        # previous moves (for the opponent's MCTS)
        self.previous_moves = (None, None)

        self.GUI = GUI(800, 800)
        self.GUI.start()
        self.GUI.update(self.game.env.board.fen())


    def play_game(self):
        self.game.reset()
        # show the board
        while True:
            if self.player == self.game.turn:
                self.get_player_move()
                self.game.turn = not self.game.turn
            else:
                self.opponent_move()
            # check if the game is over
            if self.game.env.board.is_game_over():
                # get the winner
                winner = Game.get_winner(self.game.env.board.result(claim_draw=True))
                # show the winner
                print(f"\n{'White' if winner == 1 else 'Black'} wins!")
                break

    def get_player_move(self):
        while True:
            # get the move from the player
            move = input("Enter your move: ")
            # check if the move is valid
            try:
                # convert the move to a chess.Move object
                print(move)
                move = self.game.env.board.parse_san(move)
            except ValueError:
                print("Invalid move! Try again.")
                continue
            break
        # play the move
        self.game.env.board.push(move)
        print(self.game.env.board)

    def opponent_move(self):
        self.previous_moves = self.game.play_move(stochastic=False, previous_moves=self.previous_moves, save_moves=False)


if __name__ == "__main__":
    m = Main(False)
    # create new thread for game logic
    thread = threading.Thread(target=m.play_game)
    thread.start()
    while True:
        m.GUI.get_events()
