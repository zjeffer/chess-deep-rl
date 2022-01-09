# implement the Monte Carlo Tree Search algorithm
from tensorflow.python.ops.numpy_ops import np_config
import chess
import chess.pgn
from tensorflow.python.ops.numpy_ops.np_math_ops import positive
from chessEnv import ChessEnv
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

np_config.enable_numpy_behavior()


class MCTS:
    def __init__(self, agent: "Agent", state: str = chess.STARTING_FEN):
        self.root = Node(state=state)
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
        # in the first sim, the root node has no children, so leaf = root
        leaf = self.select_child(node)
        leaf.N += 1

        # expand the leaf node
        end_node = self.expand(leaf)

        # backpropagate the result
        end_node = self.backpropagate(end_node, end_node.value)

        self.amount_of_simulations += 1
        del self.game_path
        return end_node

    def select_child(self, node: Node) -> Node:
        while not node.is_leaf():
            # choose the action that maximizes Q+U
            max_edge: Edge = max(
                node.edges, key=lambda edge: edge.Q + edge.upper_confidence_bound(node.N))

            # get that actions's new node
            node = max_edge.output_node
            self.game_path.append(max_edge)
        return node

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
        probabilities = probabilities.reshape(
            config.amount_of_planes, config.n, config.n)
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
            thread = threading.Thread(
                target=self.filter_valid_move, args=(move,))
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


    def expand(self, leaf: Node) -> Node:
        """
        Expand the leaf node. Use a neural network to select the best move.
        This will generate a new state
        """
        logging.debug("Expanding...")

        # get all possible moves
        possible_actions = list(chess.Board(leaf.state).generate_legal_moves())

        if not len(possible_actions):
            # TODO: return something here instead of exception
            raise Exception("No possible moves, game is over")

        # predict p and v
        # p = array of probabilities: [0, 1] for every move (including invalid moves)
        # v = [-1, 1]
        p, v = self.agent.predict(ChessEnv.state_to_input(leaf.state))
        p, v = p[0], v[0][0]
        actions = self.probabilities_to_actions(p, leaf.state)

        logging.debug(f"Model predictions: {p}")
        logging.debug(f"Value of state: {v}")

        leaf.value = v

        # create a child node for every action
        for action in possible_actions:
            # make the move and get the new board
            new_state = leaf.step(action)
            # add a new child node with the new board, the action taken and its prior probability
            leaf.add_child(Node(new_state), action, actions[action.uci()])
            self.amount_of_expansions += 1
        return leaf

    def backpropagate(self, end_node: Node, value: float) -> Node:
        logging.debug("Backpropagation...")

        # print(self.game_path)
        # print(f"Game path length: {len(self.game_path)}")
        self.game_path.reverse()

        for edge in self.game_path:
            edge.N += 1
            edge.W += value
            edge.Q = edge.W / edge.N
        return end_node

    @staticmethod
    def get_piece_amount(board: chess.Board) -> int:
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

    def plot_node(self, dot: Digraph, node: Node):
        dot.node(f"{node.state}", f"N")
        for edge in node.edges:
            dot.edge(str(edge.input_node.state), str(
                edge.output_node.state), label=edge.action.uci())
            dot = self.plot_node(dot, edge.output_node)
        return dot

    def plot_tree(self) -> None:
        # tree plotting
        dot = Digraph(comment='Chess MCTS Tree')
        print(f"# of nodes in tree: {len(self.root.get_all_children())}")

        # recursively plot the tree
        dot = self.plot_node(dot, self.root)
        dot.save('mcts_tree.gv')
