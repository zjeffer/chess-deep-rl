import chess
from chess import Move
from collections.abc import Iterator


class Node:
	def __init__(self, state: chess.Board, parent: "Node" = None, action: Move = None):
		"""
		A node is a state inside the MCTS tree.
		"""
		self.state = state
		# the Node's parent. None if the node is the root node
		self.parent = parent
		# the move that led to this node
		self.action = action
		# the value of this node TODO: implement
		self.propagated_value: float = 0.0

		# the unexplored actions for this state
		self.unexplored_actions: list[Move] = []
		# the node's children
		self.children: list[Node] = []
		# upper confidence bound
		self.estimated_value: int = 1e6

	def get_unexplored_actions(self) -> Iterator[Move]:
		""" 
		Get all unexplored actions for the current state. Returns a generator.
		"""
		return self.state.generate_legal_moves()

	def step(self, action: Move) -> Move:
		"""
		Take a step in the game, returns the move taken or None if an error occured
		"""
		try:
			self.state.push(action)
		except ValueError:
			print("Error: Invalid move.")
			return None
		return action

	def is_game_over(self) -> bool:
		"""
		Check if the game is over.
		"""
		return self.state.is_game_over()

	def is_leaf(self) -> bool:
		"""
		Check if the current node is a leaf node.
		"""
		return self.children == []

	def select_child(self) -> "Node":
		"""
		Select a child node to expand.
		"""
		return random.choice(self.children)

	def add_child(self, child: "Node") -> "Node":
		"""
		Add a child node to the current node.
		"""
		self.children.append(child)
		return child
