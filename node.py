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

        # the explored actions for this state
        self.explored_actions: list[Move] = []
        # the node's children
        self.children: list[Node] = []
        # upper confidence bound
        self.estimated_value: int = 1e6

    def __eq__(self, node: object) -> bool:
        """
        Check if two nodes are equal.
                Two nodes are equal if the state is the same and it got there through the same action
        """
        if isinstance(node, Node):
            return self.state == node.state and self.action == node.action
        else:
            return NotImplemented

    def get_unexplored_actions(self) -> list[Move]:
        """ 
        Get all unexplored actions for the current state. Returns a generator.
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
        return not len(self.children)

    def add_child(self, child: "Node") -> "Node":
        """
        Add a child node to the current node.
        """
        self.children.append(child)
        return child

    def calculate_children(self) -> None:
        """
        Calculate the children of the current node.
        """
        for action in self.get_unexplored_actions():
            state = self.state.copy()
            state.push(action)
            self.add_child(Node(state=state, parent=self, action=action))

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
