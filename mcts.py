# implement the Monte Carlo Tree Search algorithm

import chess
import random

class MCTS:
    def __init__(self):
        self.root = Node(state=chess.Board())
        self.amount_of_simulations = 0
        
    def run_simulation(self):
        node = self.root
        # select a leaf node to expand
        leaf = self.select_child(node)
        # expand the leaf node
        actions = leaf.get_unexplored_actions()
        
        # rollout the leaf node

        # backpropagate the result

    def select_child(self, node):
        # find a leaf node
        while not node.is_leaf():
            node = random.choice(node.children)
        return node


    def expand(self):
        # add a new child to the leaf
        leaf = self.select_child(self.root)
        # take a random action
        action = random.choice(leaf.get_unexplored_actions())
        leaf.step(action)
        # get the new state
        new_state = leaf.state.copy()
        # create a new node with the new state
        leaf.add_child(Node(state=new_state, parent=leaf, action=action))
        

    def rollout(self):
        pass

    def backpropagate(self):
        pass


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