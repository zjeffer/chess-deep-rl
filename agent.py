from tensorflow.python.types.core import ConcreteFunction
from chessEnv import ChessEnv
from rlmodelbuilder import RLModelBuilder
import config
from keras.models import Model
import time
import tensorflow as tf
import utils
from tqdm import tqdm
from mcts import MCTS

class Agent:
    def __init__(self, build_model: bool = True):
        self.model: Model = None
        self.MAX_REPLAY_MEMORY = config.MAX_REPLAY_MEMORY
        self.tf_model: ConcreteFunction = None
        if build_model:
            # this if statement is useful for testing purposes
            self.model = self.build_model()

        self.mcts = MCTS(self)

        # memory
        self.memory = []

    def build_model(self) -> Model:
        # create the model
        model_builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        model = model_builder.build_model()
        return model

    def run_simulations(self, n: int = 1):
        start_time = time.time()
        print(f"Running {n} simulations...")
        # run n simulations
        self.mcts.run_simulations(n)
        print("="*50)
        print(f"Time: {(time.time() - start_time):.3f} seconds for {n} simulations")
        print("="*50)

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