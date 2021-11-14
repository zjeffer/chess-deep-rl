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
