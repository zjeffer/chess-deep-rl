from rlmodelbuilder import RLModelBuilder
import config
from keras.models import Model

class Agent:
    def __init__(self):

        # create the model
        # TODO: change the number of residual blocks if necessary
        model_builder = RLModelBuilder(
            config.INPUT_SHAPE, config.OUTPUT_SHAPE, nr_hidden_layers=config.AMOUNT_OF_RESIDUAL_BLOCKS)
        self.model: Model = model_builder.build_model()

        # memory
        self.memory = []

    def play_one_move(self):
        pass

    def save_to_memory(self, game):
        if len(self.memory) >= self.max_replay_memory:
            self.memory.pop(0)
        self.memory.append(game)

    def evaluate_network(self, best_model, amount=400):
        """
        Test to see if new network is stronger than the current best model
        Do this by playing x games. If the new network wins more, it is the new best model 
        """
        pass