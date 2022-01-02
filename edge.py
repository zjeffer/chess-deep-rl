from chess import Move


class Edge:
    def __init__(self, input_node: "Node", output_node: "Node", action: Move, P: float):
        self.input_node = input_node
        self.output_node = output_node
        self.action = action

        self.player_turn = self.input_node.state.turn

        # each action stores 4 numbers:
        self.N = 0  # amount of times this action has been taken
        self.W = 0  # total value of next state
        self.Q = 0  # mean value of next state
        self.P = P  # prior probability of selecting this action

    def __eq__(self, edge: object) -> bool:
        if isinstance(edge, Edge):
            return self.action == edge.action
        else:
            return NotImplemented

    def upper_confidence_bound(self) -> float:
        # TODO: implement
        pass
