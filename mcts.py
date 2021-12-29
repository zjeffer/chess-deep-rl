# implement the Monte Carlo Tree Search algorithm

import chess
from chessEnv import ChessEnv
import random
from node import Node


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
		actions = leaf.get_unexplored_actions()

		# rollout the leaf node
		end_node = self.rollout(leaf)

		# backpropagate the result
		self.backpropagate(end_node)

		self.amount_of_simulations += 1

	def select_child(self, node) -> Node:
		print("Getting leaf node...")
		# find a leaf node
		while not node.is_leaf():
			# TODO: choose the action that maximizes Q+U
			# Q = value of the next state
			# U = function of P (prior prob) 
			# 	  and N (amount of times the action has been taken in current state)

			# TODO: for now, just select a random child
			node = random.choice(node.children)
		return node

	def expand(self, leaf: Node) -> Node:
		print("Expanding...")
		self.amount_of_expansions += 1
		actions = list(leaf.get_unexplored_actions())
		# take a random action
		action = random.choice(actions)
		# don't update the leaf node's state, just the child's state
		old_state = leaf.state.copy()
		# make the move
		next_move = leaf.step(action)

		# create a new node
		if next_move is None:
			# TODO
			print("ERROR: Invalid move")

		# create a new node with the new state
		new_node: Node = leaf.add_child(
			Node(state=leaf.state.copy(), parent=leaf, action=action))
		leaf.state = old_state
		return new_node

	def rollout(self, node: Node) -> Node:
		print("Rolling out...")
		# node is the current node
		while not node.is_game_over():
			# calculate children for the current node
			node = self.expand(node)
		print("Rollout finished!")
		return node
		

	def backpropagate(self, end_node):
		print("Backpropagation...")
		# TODO: implement
		from chess import Move

		print(end_node.state)
		print(end_node.state.result())
		game = chess.pgn.Game.from_board(board=end_node.state)
		print(game)
		
		print("Backpropagation finished!")


