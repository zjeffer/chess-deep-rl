import base64
import logging
import socket
from rlmodelbuilder import RLModelBuilder
import config
from keras.models import Model
import time
# import tensorflow as tf
import utils
from tqdm import tqdm
from mcts import MCTS
# from tensorflow.keras.models import load_model
import requests
import json
import numpy as np

logging.basicConfig(level=logging.INFO, format=' %(message)s')

url = "http://localhost:5000/predict"

class Agent:
    def __init__(self, model_path: str = None):
        self.MAX_REPLAY_MEMORY = config.MAX_REPLAY_MEMORY

        if model_path is None:
            self.model: Model = self.build_model()
        else:
            #self.model = load_model(model_path)
            pass

        self.mcts = MCTS(self)

        # memory
        self.memory = []

        # connect to the server to do predictions
        self.socket_to_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_to_server.connect((config.SOCKET_HOST, config.SOCKET_PORT))
        logging.info(f"Agent connected to server {config.SOCKET_HOST}:{config.SOCKET_PORT}")

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

    def save_model(self, timestamped: bool = False):
        """
        Save the model to a file
        """
        if timestamped:
            self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.h5")
        else:
            self.model.save(f"{config.MODEL_FOLDER}/model.h5")

    def predict(self, data):
        """
        Send data to the server and get the prediction
        """
        # send data to server
        self.socket_to_server.send(data)
        # get msg length
        data_length = self.socket_to_server.recv(10)
        data_length = int(data_length.decode("ascii"))
        # get prediction
        response = utils.recvall(self.socket_to_server, data_length)
        # decode response
        response = response.decode("ascii")
        # json to dict
        response = json.loads(response)
        # unpack dictionary to tuple
        return np.array(response["prediction"]), response["value"]


    # @tf.function
    # def predict(self, args):
    #     if hasattr(self, 'strategy'):
    #         return self.strategy.run(self.pred_fn, args=(args,))
    #     return self.model(args)

    # def pred_fn(self, args):
    #     return self.model(args)
