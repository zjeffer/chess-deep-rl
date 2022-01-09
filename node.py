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
        # the edges connected to this node
        self.edges: list[Edge] = []
        # visit count
        self.N = 0

        # result of this node. 1 = won, 0 = draw, -1 = lost, None = not done yet
        self.result = None

        # the explored actions for this state
        self.explored_actions: list[Move] = []

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

    def get_unexplored_actions(self) -> list[Move]:
        """ 
        Get all unexplored actions for the current state. Remove already explored actions
        """
        board = chess.Board(self.state)
        actions = list(board.generate_legal_moves())
        for a in self.explored_actions:
            actions.remove(a)
        del board
        return actions

    def step(self, action: Move) -> str:
        """
        Take a step in the game, returns new state
        """
        board = chess.Board(self.state)
        board.push(action)
        new_state = board.fen()
        del board
        self.explored_actions.append(action)
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

    def add_child(self, child: "Node", action: Move, prior: float) -> Edge:
        """
        Add a child node to the current node.

        Returns the created edge between the nodes
        """
        edge = Edge(input_node=self, output_node=child, action=action, prior=prior)
        self.edges.append(edge)
        return edge

    def get_all_children(self) -> list["Node"]:
        """
        Get all children of the current node and their children, recursively
        """
        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.extend(edge.output_node.get_all_children())
        return children

    def estimate_score(self) -> int:
        """
        Estimate the score of the current node.
        Pawn = 1, Bishop = 3, Rook = 5, Queen = 9
        Positive score = white wins, negative score = black wins
        """
        score = 0
        piece_scores = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        for piece in self.state.piece_map().values():
            if piece.color == chess.WHITE:
                score += piece_scores[piece.piece_type]
            else:
                score -= piece_scores[piece.piece_type]
        return score
