from agent import Agent
from chessEnv import ChessEnv
from game import Game
import utils
import logging
import numpy as np
import selfplay



class Test:
    """
    Class to do unit tests and benchmark functions for optimization.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # don't build the model (it's not needed for these tests)
        white = Agent()
        black = Agent()
        self.game = Game(ChessEnv(), white, black)

    @utils.time_function
    def run_state_to_input_test(self, n: int = 1):
        for _ in range(n):

            self.reset()
            # test en passant
            for move in ["e4", "a5", "e5", "Nc6", "d4", "d5"]:
                self.env.step(move)
            
            

            # test input_state
            input_state = self.env.state_to_input(self.env.board)

        # names = ['white_turn', 'black_turn',
        # 	'castling_king_white', 'castling_king_black', 'castling_queen_white', 'castling_queen_black',
        # 	'is_repitition',
        # 	'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens', 'white_king',
        # 	'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens', 'black_king',
        # 	'en_passant']
        # utils.save_input_state_to_imgs(input_state, 'tests/input_planes', names)

    @utils.time_function
    def test_mask1(self):
        for _ in range(50):
            mask = np.asarray([0 for _ in range(64*73)]).reshape(73, 8, 8)
    
    @utils.time_function
    def test_mask2(self):
        for _ in range(50):
            mask = np.zeros(64*73).reshape(73, 8, 8)
    
    @utils.time_function
    def test_mask3(self):
        for _ in range(50):
            mask = np.asarray([np.asarray([0 for _ in range(64)]).reshape(8, 8) for _ in range(73)])
    
    @utils.time_function
    def test_mcts_tree(self, n: int):
        game = selfplay.setup()
        game.white.run_simulations(n)

        # get height of tree
        print(f"Tree height: {utils.get_height_of_tree(game.white.mcts.root)}")

        # plot tree
        game.white.mcts.plot_tree(f"tests/mcts_tree_{n}_nodes.gv")
        

if __name__ == "__main__":
    test = Test()
    # test.run_state_to_input_test(n=20)
    # test.test_mask1()
    # test.test_mask2()
    # test.test_mask3()

    test.test_mcts_tree(20)
    test.test_mcts_tree(400)
    test.test_mcts_tree(1200)
