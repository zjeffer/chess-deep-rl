from agent import Agent
from chessEnv import ChessEnv
from game import Game
import utils
import logging
import numpy as np
import selfplay
import chess


class Test:
    """
    Class to do unit tests and benchmark functions for optimization.
    """
    def __init__(self):
        pass


    @utils.time_function
    def run_state_to_input_test(self):
        board = chess.Board()
        # test en passant
        for move in ["e4", "a5", "e5", "Nc6", "d4", "d5"]:
            board.push_san(move)

        # test input_state
        input_state = ChessEnv.state_to_input(board.fen())
        # reshape
        input_state = np.reshape(input_state, (19, 8, 8))
        print(input_state.shape)
        
        utils.save_input_state_to_imgs(input_state, 'tests/input_planes')

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
        
    @utils.time_function
    def test_position_outputs(self, position: str = chess.STARTING_FEN, n: int = 50):
        game = selfplay.setup(position)
        
        

        # get playing side's agent
        agent = game.white if game.env.board.turn == chess.WHITE else game.black
        # create a tree by running sims
        agent.run_simulations(n)

        print(f"Input: {game.env.board.fen()}")

        # print outputs
        print("Outputs: ")
        print(f"Visit count for the root node: {agent.mcts.root.N}")
        for action in sorted(agent.mcts.root.edges, key=lambda edge: (edge.W/(edge.N if edge.N != 0 else 1))+edge.upper_confidence_bound(), reverse=True):
            print(action)



if __name__ == "__main__":
    test = Test()
    test.run_state_to_input_test()
    # test.test_mask1()
    # test.test_mask2()
    # test.test_mask3()

    # test.test_mcts_tree(20)
    # test.test_mcts_tree(400)
    # test.test_mcts_tree(1200)

    # test.test_position_outputs("1k6/1pp5/p3B2p/3Pq3/2P1p3/PP3r2/4Q3/5RK1 b - - 0 36", 400)
