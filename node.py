import chess
from chess import Move
from collections.abc import Iterator
from edge import Edge


class Node:
    def __init__(self, state: chess.Board):
        """
        A node is a state inside the MCTS tree.
        """
        self.state = state
        # the edges connected to this node
        self.edges: list[Edge] = []
        # result of this node. 1 = won, 0 = draw, -1 = lost, None = not done yet
        self.result = None

        # the explored actions for this state
        self.explored_actions: list[Move] = []

        self.value = 0

    def __eq__(self, node: object) -> bool:
        """
        Check if two nodes are equal.
        Two nodes are equal if the state is the same and it got there through the same action
        """
        if isinstance(node, Node):
            # TODO: is this the correct way to compare boards?
            return self.state == node.state and self.state.move_stack[-1] == node.state.move_stack[-1]
        else:
            return NotImplemented

    def get_unexplored_actions(self) -> list[Move]:
        """ 
        Get all unexplored actions for the current state. Remove already explored actions
        """
        actions = list(self.state.generate_legal_moves())
        for a in self.explored_actions:
            actions.remove(a)
        return actions

    def step(self, action: Move) -> Move:
        """
        Take a step in the game, returns the move taken or None if an error occured
        """
        try:
            self.state.push(action)
            self.explored_actions.append(action)
        except ValueError:
            print("ERROR: Invalid move.")
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
        return not len(self.edges)

    def add_child(self, child: "Node", action: Move, prior: float) -> Edge:
        """
        Add a child node to the current node.

        Returns the created edge between the nodes
        """
        edge = Edge(input_node=self, output_node=child, action=action, prior=prior)
        self.edges.append(edge)
        return edge

    def calculate_children(self) -> None:
        """
        Calculate the children of the current node.
        """
        for action in self.get_unexplored_actions():
            state = self.state.copy()
            state.push(action)
            self.add_child(Node(state=state))

    def get_all_children(self) -> list["Node"]:
        """
        Get all children of the current node and their children, recursively
        """
        if len(self.children) == 0:
            return []
        children = []
        for child in self.children:
            children.append(child)
            children.extend(child.get_all_children())
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
