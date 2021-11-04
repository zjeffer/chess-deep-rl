import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import optimizers
from tensorflow.keras.layers import add as add_layer


class RLModel:
    """ 
    This class builds the neural network architecture according to the AlphaGo paper.
    """

    def __init__(self, inputShape: tuple, outputShape: tuple):
        """ 
        A neural network f that takes as input the raw board presentation s of the position and its history. 
        It outputs move probabilities p and a value v:
        (p, v) = f(s)

        * p represents the probability of selecting each move.
        * v represents the probability of the current player winning the game in position s.
        """
        # define class variables
        self.inputShape = inputShape
        self.outputShape = outputShape
        # build the model
        self.model = build_model()

    def build_model(self):
        """
        Builds the neural network architecture
        """
        main_input = Input(shape=self.inputShape, name='main_input')

        x = self.conv_layer(main_input)

        # TODO: add a high amount of hidden layers
        #
        #

        policy_head = self.build_policy_head()
        value_head = self.build_value_head()

        model = Model(inputs=main_input, outputs=[
                      policy_head(x), value_head(x)])
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

    def build_convolutional_layer(self, input_layer):
        """
        Builds a convolutional layer
        """
        layer = Conv2D(filters=256, kernel_size=(3, 3), strides=(
            1, 1), padding='same', data_format='channels_first', use_bias=False)(input_layer)
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)
        return (layer)

    def build_residual_layer(self, input_layer):
        """
        Builds a residual layer
        """
        # first convolutional layer
        layer = self.build_convolutional_layer(input_layer)
        # second convolutional layer with skip connection
        layer = Conv2D(filters=256, kernel_size=(3, 3), strides=(
            1, 1), padding='same', data_format='channels_first', use_bias=False)(layer)
        layer = BatchNormalization(axis=1)(layer)
        # skipp connection
        layer = add_layer([layer, input_layer])
        layer = LeakyReLU()(layer)

        return (layer)

    def build_policy_head(self):
        """
        Builds the policy head of the neural network
        """
        model = Sequential()
        model.add(Conv2D(2, kernel_size=(1, 1), strides=(
            1, 1), input_shape=self.inputShape, padding='same', data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(self.outputShape[0], name='policy_head'))
        return model

    def build_value_head(self):
        """
        Builds the value head of the neural network
        """
        model = Sequential()
        model.add(Conv2D(1, kernel_size=(1, 1), strides=(
            1, 1), input_shape=self.inputShape, padding='same', data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(20))
        model.add(LeakyReLU())
        # output shape == 1, because we want 1 value: the probability of the current player winning the game
        # tanh activation function maps the output to [-1, 1]
        model.add(Dense(1, activation='tanh', name='value_head'))
        return model
