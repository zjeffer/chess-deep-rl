# implement the Monte Carlo Tree Search algorithm
import chess
import chess.pgn
from chessEnv import ChessEnv
from agent import Agent
from node import Node
from edge import Edge
import numpy as np
import time
import tqdm
import utils
import threading

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
    def __init__(self, agent: Agent):
        self.root = Node(state=chess.STARTING_FEN)
        self.amount_of_simulations = 0
        self.amount_of_expansions = 0

        self.game_path: list[Edge] = []
        self.cur_board: chess.Board = None

        self.agent = agent

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
        del self.game_path
        return end_node

    def filter_valid_move(self, move) -> None:
        logging.debug("Filtering valid moves...")
        from_square = move.from_square
        to_square = move.to_square

        plane_index: int = None
        piece = self.cur_board.piece_at(from_square)
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
        self.outputs.append((move, plane_index, row, col))

    @utils.timer_function
    def probabilities_to_actions(self, probabilities: list, board: str) -> dict:
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
        probabilities = probabilities.reshape(config.amount_of_planes, config.n, config.n)
        mask = np.zeros((config.amount_of_planes, config.n, config.n))

        actions = {}

        # only get valid moves
        self.cur_board = chess.Board(board)
        valid_moves = self.cur_board.generate_legal_moves()
        self.outputs = []
        threads = []
        while True:
            try:
                move = next(valid_moves)
            except StopIteration:
                break
            thread = threading.Thread(target=self.filter_valid_move, args=(move,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        probabilities = probabilities.numpy()
        for move, plane_index, col, row in self.outputs:
            mask[plane_index][col][row] = 1
            actions[move.uci()] = probabilities[plane_index][col][row]

        # utils.save_output_state_to_imgs(mask, "tests/output_planes", "mask")
        # utils.save_output_state_to_imgs(probabilities, "tests/output_planes", "unfiltered")

        # use the mask to filter the probabilities
        probabilities = np.multiply(probabilities, mask)

        # utils.save_output_state_to_imgs(probabilities, "tests/output_planes", "filtered")
        return actions

    @utils.timer_function
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

        while self.amount_of_expansions < config.MAX_DEPTH:
            # print(f"{self.amount_of_expansions} expansions, move_stack length: {len(leaf.state.move_stack)}")
            self.amount_of_expansions += 1
            # don't update the leaf node's state, just the child's state
            state = leaf.state

            # predict p and v
            # p = array of probabilities: [0, 1] for every move (including invalid moves)
            # v = [-1, 1]
            p, v = self.agent.predict(ChessEnv.state_to_input(state))
            p, v = p[0], v[0][0]

            logging.debug(f"Model predictions: {p}")
            logging.debug(f"Value of state: {v}")
            
            actions = self.probabilities_to_actions(p, state)

            if not len(actions):
                logging.debug("No valid moves, stopping expansion")
                return leaf

            # get action with highest probability
            max_action = max(actions, key=lambda action: actions[action])
            logging.debug(f"Best action: {max_action}")

            # make the move. this changes leaf.state
            new_state = leaf.step(max_action)

            # create new node
            new_node = Node(state=new_state)
            new_node.value = v
            # set the state back to the old one (undo the move)
            leaf.state = state
            # create the edge between this node and the new node
            edge = leaf.add_child(new_node, max_action, actions[max_action])
            # add the edge to the tree
            self.game_path.append(edge)
            
            # new node is now leaf node
            leaf = new_node

        return leaf

    @utils.timer_function
    def backpropagate(self, end_node: Node, value: float):
        logging.debug("Backpropagation...")

        self.game_path.reverse()

        for edge in self.game_path:
            edge.N += 1
            edge.W += value
            edge.Q = edge.W / edge.N
        # print(f"Q: {self.game_path[0].Q}, \t U:{self.game_path[0].upper_confidence_bound(self.amount_of_simulations)}")
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
