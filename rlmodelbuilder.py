import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, BatchNormalization, LeakyReLU, Input
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add as add_layer
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


class RLModelBuilder:
    """ 
    This class builds the neural network architecture according to the AlphaGo paper.
    """

    def __init__(self, input_shape: tuple, output_shape: tuple, nr_hidden_layers: int = 0):
        """ 
        A neural network f that takes as input the raw board presentation s of the position and its history. 
        It outputs move probabilities p and a value v:
        (p, v) = f(s)

        * p represents the probability of selecting each move.
        * v represents the probability of the current player winning the game in position s.
        """
        # define class variables
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nr_hidden_layers = nr_hidden_layers

    def build_model(self) -> Model:
        """
        Builds the neural network architecture
        """
        main_input = Input(shape=self.input_shape, name='main_input')

        x = self.build_convolutional_layer(main_input)

        # add a high amount of residual layers
        for i in range(self.nr_hidden_layers):
            x = self.build_residual_layer(x)

        model = Model(inputs=main_input, outputs=x)
        return model

        policy_head = self.build_policy_head()
        value_head = self.build_value_head()

        model = Model(inputs=main_input,
                      outputs=[policy_head(x), value_head(x)])

        model.compile(
            loss={
                # TODO: change to better (own) loss function
                'policy_head': 'categorical_crossentropy',
                'value_head': 'mean_squared_error'
            },
            optimizer=optimizers.Adam(lr=0.001),
            loss_weights={
                'policy_head': 0.5,
                'value_head': 0.5
            }
        )
        # return the compiled model
        return model

    def build_convolutional_layer(self, input_layer) -> KerasTensor:
        """
        Builds a convolutional layer
        """

        # TODO: change parameters for these layers (data_format, etc)
        layer = Conv2D(filters=256, kernel_size=(3, 3), strides=(
            1, 1), padding='same', data_format='channels_first', use_bias=False)(input_layer)
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)
        return (layer)

    def build_residual_layer(self, input_layer) -> KerasTensor:
        """
        Builds a residual layer
        """
        # first convolutional layer
        layer = self.build_convolutional_layer(input_layer)
        # second convolutional layer with skip connection
        layer = Conv2D(filters=256, kernel_size=(3, 3), strides=(
            1, 1), padding='same', data_format='channels_first', use_bias=False)(layer)
        layer = BatchNormalization(axis=1)(layer)
        # skip connection
        layer = add_layer([layer, input_layer])
        # activation function
        layer = LeakyReLU()(layer)
        return (layer)

    def build_policy_head(self) -> Model:
        """
        Builds the policy head of the neural network
        """
        model = Sequential()
        model.add(Conv2D(256, kernel_size=(1, 1), strides=(
            1, 1), input_shape=self.input_shape, padding='same', data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        # according to alphazero paper: 73 filters for chess
        model.add(Conv2D(73, kernel_size=(1, 1)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(self.output_shape[0], name='policy_head'))
        return model

    def build_value_head(self) -> Model:
        """
        Builds the value head of the neural network
        """
        model = Sequential()
        model.add(Conv2D(1, kernel_size=(1, 1), strides=(
            1, 1), input_shape=self.input_shape, padding='same', data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(20))
        model.add(LeakyReLU())
        # output shape == 1, because we want 1 value: the estimated outcome from the position
        # tanh activation function maps the output to [-1, 1]
        model.add(Dense(self.output_shape[1],
                  activation='tanh', name='value_head'))
        return model
