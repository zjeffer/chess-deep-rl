# implement the Monte Carlo Tree Search algorithm
import chess
import chess.pgn
from chessEnv import ChessEnv
from node import Node
from edge import Edge
import numpy as np
import time
import tqdm
import utils

# graphing mcts
from graphviz import Digraph

import config
# output vector mapping
from mapper import Mapping

import logging
logging.basicConfig(level=logging.DEBUG)

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class MCTS:
    def __init__(self, env: ChessEnv):
        self.env = env
        self.root = Node(state=self.env.board)
        self.amount_of_simulations = 0
        self.amount_of_expansions = 0

        self.game_path: list[Edge] = []

    @utils.timer_function
    def run_simulation(self):
        self.game_path = []
        self.amount_of_expansions = 0
        node = self.root
        # traverse the tree by selecting edges with max Q+U
        leaf = self.select_child(node)

        # expand the leaf node (recursive function)
        end_node = self.expand(leaf)

        # backpropagate the result
        end_node = self.backpropagate(end_node, end_node.value)

        self.amount_of_simulations += 1
        return end_node

    def probabilities_to_actions(self, probabilities: list, board: chess.Board) -> dict:
        """
        Map the output vector of 4672 probabilities to moves

        The output vector is a list of probabilities for every move
        * 4672 probabilities = 73*64 => 73 planes of 8x8

        The squares in these 8x8 planes indicate the square where the piece is.

        The plane itself indicates the type of move:
            - first 56 planes: queen moves (length of 7 squares * 8 directions)
            - next 8 planes: knight moves (8 directions)
            - final 9 planes: underpromotions (left diagonal, right diagonal, forward) * (three possible pieces (knight, bishop, rook))
        """
        probabilities = probabilities.reshape(73, 8, 8)

        actions = {}

        # only get valid moves
        valid_moves = list(board.generate_legal_moves())
        logging.debug(f"Amount of valid moves: {len(valid_moves)}")

        mask = np.asarray([np.asarray([0 for _ in range(64)]).reshape(
            8, 8) for _ in range(config.amount_of_planes)])
        logging.debug(f"Mask shape: {mask.shape}")
        for move in valid_moves:
            from_square = move.from_square
            to_square = move.to_square

            plane_index: int = None
            piece = board.piece_at(from_square)
            direction = None

            if piece is None:
                raise Exception(f"No piece at {from_square}")

            if move.promotion and move.promotion != chess.QUEEN:
                piece_type, direction = Mapping.get_underpromotion_move(
                    move.promotion, from_square, to_square)
                plane_index = Mapping.mapper[piece_type][1 - direction]
            else:
                # find the correct plane based on from_square and move_square
                if piece.piece_type == chess.KNIGHT:
                    # get direction
                    direction = Mapping.get_knight_move(from_square, to_square)
                    plane_index = Mapping.mapper[direction]
                else:
                    # get direction of queen-type move
                    direction, distance = Mapping.get_queenlike_move(
                        from_square, to_square)
                    plane_index = Mapping.mapper[direction][np.abs(distance)-1]
            # create a mask with only valid moves
            row = from_square % 8
            col = 7 - (from_square // 8)
            mask[plane_index][col][row] = 1
            actions[move] = probabilities[plane_index][col][row]

        # utils.save_output_state_to_imgs(mask, "tests/output_planes", "mask")
        # utils.save_output_state_to_imgs(probabilities, "tests/output_planes", "unfiltered")

        # use the mask to filter the probabilities
        probabilities = np.multiply(probabilities, mask)

        # utils.save_output_state_to_imgs(probabilities, "tests/output_planes", "filtered")

        # probabilities to dictionary with actions

        return actions

    def select_child(self, node: Node) -> Node:
        logging.debug("Getting leaf node...")
        # find a leaf node
        while not node.is_leaf():
            logging.debug("Getting random child...")
            # choose the action that maximizes Q+U
            max_edge: Edge = max(node.edges, key=lambda edge: edge.Q +
                                 edge.upper_confidence_bound(self.amount_of_simulations))

            # get the child node with the highest Q+U
            node = max_edge.output_node
        return node

    @utils.timer_function
    def expand(self, leaf: Node) -> Node:
        """
        Expand the leaf node. Use a neural network to select the best move.
        This will generate a new state
        """
        logging.debug("Expanding...")

        # print(f"{self.amount_of_expansions} expansions, move_stack length: {len(leaf.state.move_stack)}")
        self.amount_of_expansions += 1
        # don't update the leaf node's state, just the child's state
        state = leaf.state.copy()

        # predict p and v
        # p = array of probabilities: [0, 1] for every move (including invalid moves)
        # v = [-1, 1]
        start_time = time.time()
        if state.turn:
            p, v = self.env.black.predict(ChessEnv.state_to_input(state))
        else:
            p, v = self.env.black.predict(ChessEnv.state_to_input(state))
        p, v = p[0], v[0]
        print(f"Prediction time: {time.time() - start_time}")

        logging.debug(f"Model predictions: {p}, {v}")
        logging.debug(f"Value of state: {v}")

        
        actions = self.probabilities_to_actions(p, state)

        if not len(actions):
            print("End of recursion")
            return leaf
        logging.debug(len(actions))

        # get action with highest probability
        max_action = max(actions, key=lambda action: actions[action])

        # make the move. this changes leaf.state
        leaf.step(max_action)

        # create new node
        new_node = Node(state=leaf.state.copy())
        new_node.value = v
        # set the state back to the old one (undo the move)
        leaf.state = state
        # create the edge between this node and the new node
        edge = leaf.add_child(new_node, max_action, actions[max_action])
        # add the edge to the tree
        self.game_path.append(edge)
        # update the value of the new leaf node
        

        # recursion: expand the new node
        if self.amount_of_expansions >= 50:
            new_node.result = 0
            return new_node
        return self.expand(new_node)

    def backpropagate(self, end_node: Node, value: float):
        logging.debug("Backpropagation...")
        start_time = time.time()

        self.game_path.reverse()
        for edge in self.game_path:
            edge.N += 1
            edge.W += value
            edge.Q = edge.W / edge.N
            print(edge)

        logging.debug("Backpropagation finished in " + str(time.time() - start_time) + " seconds")
        return end_node

    @staticmethod
    def get_piece_amount(board: chess.Board):
        return len(board.piece_map().values())

    def estimate_winner(self, node: Node) -> int:
        score = node.estimate_score()
        if np.abs(score) > 1:
            if score > 0:
                logging.debug("White wins")
                return 1
            else:
                logging.debug("Black wins")
                return -1
        else:
            logging.debug("Draw")
            return 0

    def plot_tree(self):
        # tree plotting
        # TODO: fix because i'm now using an Edge class
        dot = Digraph(comment='Chess MCTS Tree')
        print(f"# of nodes in tree: {len(self.root.get_all_children())}")
        print(f"Plotting tree...")
        for node in tqdm(self.root.get_all_children()):
            dot.node(str(node.state.fen()), label="*")
            for child in node.children:
                dot.edge(str(node.state.fen()), str(
                    child.state.fen()), label=str(child.action))
        dot.save('mcts_tree.gv')
