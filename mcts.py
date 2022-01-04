# implement the Monte Carlo Tree Search algorithm

from xml.etree.ElementTree import tostring
import chess
import chess.pgn
import random

from chessEnv import ChessEnv
from node import Node
from edge import Edge
import numpy as np
import time
import tqdm

# graphing mcts
from graphviz import Digraph

import config
# output vector mapping
from mapper import KnightMove, Mapping, QueenDirection, UnderPromotion

import logging
logging.basicConfig(level=logging.INFO)


class MCTS:
    def __init__(self, env: ChessEnv):
        self.env = env
        self.root = Node(state=self.env.board)
        self.amount_of_simulations = 0
        self.amount_of_expansions = 0

        self.game_path: list[Node] = []

    def run_simulation(self):
        node = self.root
        # traverse the tree by selecting edges with max Q+U
        leaf = self.select_child(node)

        # expand the leaf node. evaluate the state of the new node using the NN
        new_node = self.expand(leaf)

        # rollout the new node
        end_node = self.rollout(new_node)

        # backpropagate the result
        self.backpropagate(end_node)

        self.amount_of_simulations += 1

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
        queen_moves, knight_moves, underpromotions = np.split(
            probabilities, [config.queen_planes*64, config.queen_planes*64+config.knight_planes*64])
        print(f"Amount of queen moves: {len(queen_moves)}")
        print(f"Amount of knight moves: {len(knight_moves)}")
        print(f"Amount of underpromotions: {len(underpromotions)}")

        queen_moves = np.reshape(queen_moves, (config.queen_planes, 64))
        knight_moves = np.reshape(knight_moves, (config.knight_planes, 64))
        underpromotions = np.reshape(
            underpromotions, (config.underpromotion_planes, 64))

        # only get valid moves
        valid_moves = list(self.env.board.generate_legal_moves())
        print(f"Amount of valid moves: {len(valid_moves)}")

        mask = np.asarray([np.asarray([0 for _ in range(64)]).reshape(
            8, 8) for _ in range(config.amount_of_planes)])
        print(f"Mask shape: {mask.shape}")
        for move in valid_moves:
            print(move)
            from_square = move.from_square
            to_square = move.to_square
            # print(f"From {from_square} to: {to_square}, diff: {from_square - to_square}")

            plane_index: int = None
            piece = board.piece_at(from_square)
            direction = None

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
            # based on the plane index, set the from square to true
            print(f"Plane index: {plane_index}")
            # mask invalid moves

    def select_child(self, node: Node) -> Node:
        logging.debug("Getting leaf node...")
        # find a leaf node
        while not node.is_leaf():
            logging.debug("Getting random child...")
            # choose the action that maximizes Q+U
            max_edge: Edge = max(node.edges, key=lambda edge: edge.Q + edge.U)
            max_edge.N += 1

            # get the child node with the highest Q+U
            node = max_edge.output_node

            # TODO: for now, just select a random child
            # calculate new children for the current node
            node.calculate_children()
            node = np.random.choice(node.children)
        return node

    def expand(self, leaf: Node) -> Node:
        """
        Expand the leaf node. Use a neural network to select the best move.
        This will generate a new state
        """
        logging.debug("Expanding...")
        self.amount_of_expansions += 1
        # don't update the leaf node's state, just the child's state
        old_state = leaf.state.copy()

        # predict p and v
        # p = array of probabilities: [0, 1] for every move (including invalid moves)
        # v = [-1, 1]
        if old_state.turn:
            p, v = self.env.white.model.predict(
                ChessEnv.board_to_state(old_state))
        else:
            p, v = self.env.black.model.predict(
                ChessEnv.board_to_state(old_state))
        p, v = p[0], v[0]

        print(p, v)
        print(np.mean(p)*255)
        print(f"Amount of probabilities: {len(p)}")
        print(f"Value of state: {v}")

        # test: show output with planes
        # import utils
        # p_output = p.reshape(73, 8, 8)
        # utils.save_output_state_to_imgs(p_output, "tests/output_planes")

        # TODO: map actions to probabilities
        start_time = time.time()
        p = self.probabilities_to_actions(p, old_state)
        print(f"Time to map probabilities: {time.time() - start_time}")

        exit(1)

        best_edge = None

        # update the best edge
        best_edge.W += v
        best_edge.Q = best_edge.W / best_edge.N

        # update all edges
        for edge in leaf.edges:
            # TODO: for every edge, update the prior using p.
            pass

        # make the move. this changes leaf.state
        leaf.step(action)

        # create a new node with the new state
        new_node: Node = leaf.add_child(Node(state=leaf.state.copy()))
        leaf.state = old_state
        return new_node

    def rollout(self, node: Node) -> Node:
        logging.debug("Rolling out...")
        # node is the current node
        while not node.is_game_over():
            # calculate children for the current node
            node = self.expand(node)
            self.game_path.append(node)

            # if amount of pieces on board is less than 8, consult tablebase
            if MCTS.get_piece_amount(node.state) < 8:
                logging.debug("Less than 8 pieces on board")
                # decide who wins by estimating the score for this node
                # TODO: tablebase instead of estimation
                winner = self.estimate_winner(node)
                if node.state.turn == chess.WHITE:
                    node.result = winner
                else:
                    node.result = -winner
                break

            # if move amount is higher than 100, draw
            if node.state.fullmove_number > 10:
                logging.debug("Move amount is higher than 100")
                # decide who wins by estimating the score for this node
                winner = self.estimate_winner(node)
                if node.state.turn == chess.WHITE:
                    node.result = winner
                else:
                    node.result = -winner
                break
        logging.debug("Rollout finished!")
        return node

    def backpropagate(self, end_node: Node, value: float):
        logging.debug("Backpropagation...")
        # TODO: implement

        self.game_path.reverse()
        for node in self.game_path:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N

        logging.debug(end_node.state)
        game = chess.pgn.Game.from_board(board=end_node.state)
        logging.debug(game)
        logging.debug(end_node.state.result())
        logging.debug("Backpropagation finished!")

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
