import os
import time
from typing import Tuple
import chess
import numpy as np
from chessEnv import ChessEnv
import config
import tensorflow as tf
from keras.models import Model
from keras.models import load_model, save_model
from matplotlib import pyplot as plt
import pandas as pd
import uuid
import utils
from tqdm import tqdm
from rlmodelbuilder import RLModelBuilder

class Trainer:
    def __init__(self, model: Model):
        self.model = model
        self.batch_size = config.BATCH_SIZE

    def sample_batch(self, data):
        if self.batch_size > len(data):
            return data
        else:
            np.random.shuffle(data)
            return data[:self.batch_size]

    def split_Xy(self, data) -> Tuple[np.ndarray, np.ndarray]:
        # board to input format (19x8x8)
        X = np.array([ChessEnv.state_to_input(i[0])[0] for i in data])
        # moves to output format (73x8x8)
        y_probs = []
        # values = winner
        y_value = []
        for position in data:
            # for every position in the batch, get the output probablity vector and value of the state
            board = chess.Board(position[0])
            moves = utils.moves_to_output_vector(position[1], board)
            y_probs.append(moves)
            y_value.append(position[2])
        return X, (np.array(y_probs).reshape(len(y_probs), 4672), np.array(y_value))

    def train_batch(self, X, y_probs, y_value):
        return self.model.train_on_batch(x=X, y={
                "policy_head": y_probs,
                "value_head": y_value
            }, return_dict=True)

    @utils.timer_function
    def train_model(self, data):
        """
        Train the model on batches of data

        X = the state of the board (a fen string)
        y = the search probs by MCTS (array of dicts), and the winner (-1, 0, 1)
        """
        history = []
        for _ in tqdm(range(int(len(data)/self.batch_size))):
            batch = self.sample_batch(data)
            X, (y_probs, y_value) = self.split_Xy(batch)

            losses = self.train_batch(X, y_probs, y_value)
            history.append(losses)
        
        # save the new model
        save_model(self.model, os.path.join(config.MODEL_FOLDER, "model.h5"))
        return history

    def plot_loss(self, history):
        df = pd.DataFrame(history)
        df[['loss', 'policy_head_loss', 'value_head_loss']] = df[['loss', 'policy_head_loss', 'value_head_loss']].apply(pd.to_numeric, errors='coerce')
        total_loss = df[['loss']].values
        policy_loss = df[['policy_head_loss']].values
        value_loss = df[['value_head_loss']].values
        plt.plot(total_loss, label='loss')
        plt.plot(policy_loss, label='policy_head_loss')
        plt.plot(value_loss, label='value_head_loss')
        plt.legend()
        plt.title(f"Loss over time\nLearning rate: {config.LEARNING_RATE}")
        plt.savefig(f'{config.LOSS_PLOTS_FOLDER}/loss-{str(uuid.uuid4())[:8]}.png')
        del df

if __name__ == "__main__":
    # model = load_model(os.path.join(config.MODEL_FOLDER, "model.h5"))
    model = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE).build_model()
    trainer = Trainer(model=model)

    files = os.listdir(config.MEMORY_DIR)
    data = []
    print(f"Loading all games in {config.MEMORY_DIR}...")
    for file in files:
        data.append(np.load(f"{config.MEMORY_DIR}/{file}", allow_pickle=True))
    data = np.concatenate(data)
    print(f"Training with {len(data)} positions")
    history = trainer.train_model(data)
    # plot history
    trainer.plot_loss(history)

