from tensorflow.python.types.core import ConcreteFunction
from rlmodelbuilder import RLModelBuilder
import config
from keras.models import Model
import time
import tensorflow as tf

class Agent:
    def __init__(self, build_model: bool = True):
        self.model: Model = None
        self.tf_model: ConcreteFunction = None
        if build_model:
            # this if statement is useful for testing purposes
            self.model = self.build_model()

        # memory
        self.memory = []

    def build_model(self) -> Model:
        # create the model
        model_builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        model = model_builder.build_model()

        # self.tf_model = model_builder.convert_keras_to_tensorflow_model(model)

        return model

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

    def save_model(self, timestamped: bool = False):
        """
        Save the model to a file
        """
        if timestamped:
            self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.h5")
        else:
            self.model.save(f"{config.MODEL_FOLDER}/model.h5")

    @tf.function
    def predict(self, args):
        return self.model(args)