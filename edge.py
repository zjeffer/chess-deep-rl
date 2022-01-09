from chess import Move
import config
import math
import utils

class Edge:
    def __init__(self, input_node: "Node", output_node: "Node", action: Move, prior: float):
        self.input_node = input_node
        self.output_node = output_node
        self.action = action

        self.player_turn = self.input_node.state.split(" ")[1] == "w"

        # each action stores 4 numbers:
        self.N = 0  # amount of times this action has been taken (=visit count)
        self.W = 0  # total action-value
        self.Q = 0  # mean action-value
        self.P = prior  # prior probability of selecting this action

    def __eq__(self, edge: object) -> bool:
        if isinstance(edge, Edge):
            return self.action == edge.action
        else:
            return NotImplemented

    def __str__(self):
        return f"{self.action.uci()}: Q={self.Q}, N={self.N}, W={self.W}, P={self.P}"

    def __repr__(self):
        return f"{self.action.uci()}: Q={self.Q}, N={self.N}, W={self.W}, P={self.P}"

    def upper_confidence_bound(self, N_s: int) -> float:
        exploration_rate = math.log((1 + N_s + config.C_base) / config.C_base) + config.C_init
        return exploration_rate * self.P * math.sqrt(N_s) / (1 + self.N)

    def __str__(self):
        return f"{self.action}: Q={self.Q}, N={self.N}, W={self.W}, P={self.P}"