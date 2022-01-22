import chess
from chess import Move
from collections.abc import Iterator
from edge import Edge


class Node:
    def __init__(self, state: str):
        """
        A node is a state inside the MCTS tree.
        """
        self.state = state
        self.turn = chess.Board(state).turn
        # the edges connected to this node
        self.edges: list[Edge] = []
        # the visit count for this node
        self.N = 0

        self.value = 0

    def __eq__(self, node: object) -> bool:
        """
        Check if two nodes are equal.
        Two nodes are equal if the state is the same
        """
        if isinstance(node, Node):
            return self.state == node.state
        else:
            return NotImplemented

    def step(self, action: Move) -> str:
        """
        Take a step in the game, returns new state
        """
        board = chess.Board(self.state)
        board.push(action)
        new_state = board.fen()
        del board
        return new_state

    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        """
        board = chess.Board(self.state)
        return board.is_game_over()

    def is_leaf(self) -> bool:
        """
        Check if the current node is a leaf node.
        """
        return self.N == 0

    def add_child(self, child, action: Move, prior: float) -> Edge:
        """
        Add a child node to the current node.

        Returns the created edge between the nodes
        """
        edge = Edge(input_node=self, output_node=child, action=action, prior=prior)
        self.edges.append(edge)
        return edge

    def get_all_children(self):
        """
        Get all children of the current node and their children, recursively
        """
        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.extend(edge.output_node.get_all_children())
        return children

    def get_edge(self, action) -> Edge:
        """
        Get the edge between the current node and the child node with the given action.
        """
        for edge in self.edges:
            if edge.action == action:
                return edge
        return None