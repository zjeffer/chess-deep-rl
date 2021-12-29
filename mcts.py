# implement the Monte Carlo Tree Search algorithm

import chess
import chess.pgn
import random

from chessEnv import ChessEnv
from node import Node
import numpy as np
import time

import config

import logging
logging.basicConfig(level=logging.INFO)


class MCTS:
    def __init__(self, env: ChessEnv = ChessEnv()):
        self.root = Node(state=env.board)
        self.amount_of_simulations = 0
        self.amount_of_expansions = 0

    def run_simulation(self):
        node = self.root
        # select a leaf node to expand. If the root node is a leaf node, use that as leaf
        leaf = self.select_child(node)

        # expand the leaf node
        new_node = self.expand(leaf)

        # rollout the new node
        end_node = self.rollout(new_node)

        # backpropagate the result
        self.backpropagate(end_node)

        self.amount_of_simulations += 1

    def select_child(self, node: Node) -> Node:
        logging.debug("Getting leaf node...")
        # find a leaf node
        while not node.is_leaf():
            logging.debug("Getting random child...")
            # TODO: choose the action that maximizes Q+U
            # Q = value of the next state
            # U = function of P (prior prob)
            # 	  and N (amount of times the action has been taken in current state)

            # TODO: for now, just select a random child
            # calculate new children for the current node
            node.calculate_children()
            node = np.random.choice(node.children)
        return node

    def expand(self, leaf: Node) -> Node:
        logging.debug("Expanding...")
        self.amount_of_expansions += 1
        action = random.choice(leaf.get_unexplored_actions())
        # don't update the leaf node's state, just the child's state
        old_state = leaf.state.copy()
        # make the move
        leaf.step(action)

        # create a new node with the new state
        new_node: Node = leaf.add_child(Node(leaf.state.copy(), leaf, action))
        leaf.state = old_state
        return new_node

    def rollout(self, node: Node) -> Node:
        logging.debug("Rolling out...")
        # node is the current node
        while not node.is_game_over():
            # calculate children for the current node
            node = self.expand(node)

            # if amount of pieces on board is less than 8, consult tablebase
            if MCTS.get_piece_amount(node.state) < 8:
                # TODO: tablebase
                logging.debug("Less than 8 pieces on board")
                # decide who wins by estimating the score for this node
                self.estimate_winner(node)
                break

            # if move amount is higher than 100, draw
            if node.state.fullmove_number > 10:
                # TODO: make draw
                logging.debug("Move amount is higher than 100")
                # decide who wins by estimating the score for this node
                self.estimate_winner(node)
                break
        logging.debug("Rollout finished!")
        return node

    def backpropagate(self, end_node):
        logging.debug("Backpropagation...")
        # TODO: implement

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
            # TODO: decide who wins
            if score > 0:
                logging.debug("White wins")
                score = 1
            else:
                logging.debug("Black wins")
                score = -1
        else:
            # TODO: make draw
            logging.debug("Draw")
            score = 0
        return score
