from chess import Move
import config
import math
import chess
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
        self.P = prior  # prior probability of selecting this action

    def __eq__(self, edge: object) -> bool:
        if isinstance(edge, Edge):
            return self.action == edge.action and self.input_node.state == edge.input_node.state
        else:
            return NotImplemented

    def __str__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

    def __repr__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

    def upper_confidence_bound(self) -> float:
        exploration_rate = math.log((1 + self.input_node.N + config.C_base) / config.C_base) + config.C_init
        ucb = exploration_rate * self.P * (math.sqrt(self.input_node.N) / (1 + self.N))
        if self.input_node.turn == chess.WHITE:
            return self.W / (self.N + 1) + ucb 
        else:
            return -(self.W / (self.N + 1)) + ucb
