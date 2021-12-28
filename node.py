class Node:
	def __init__(self, state: chess.Board, parent=None, action=None):
		self.state = state
		self.parent = parent
		self.action = action
		self.propagated_value = 0

		self.unexplored_actions = []
		self.children = []
		# upper confidence bound
		self.estimated_value = 1e6

	def get_unexplored_actions(self):
		""" 
		Get all unexplored actions for the current state. Returns a generator.
		"""
		return self.state.generate_legal_moves()

	def step(self, action):
		"""
		Take a step in the game, returns the move taken or None if an error occured
		"""
		try: 
			return self.state.push_uci(action)
		except ValueError:
			print("Error: Invalid move.")
		return None

	def is_game_over(self):
		"""
		Check if the game is over.
		"""
		return self.state.is_game_over()

	def is_leaf(self):
		"""
		Check if the current node is a leaf node.
		"""
		return self.children == []

	def select_child(self):
		"""
		Select a child node to expand.
		"""
		return random.choice(self.children)