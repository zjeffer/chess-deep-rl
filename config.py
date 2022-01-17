# config file: includes parameters for the model and the mcts tree

# ============= MCTS =============
SIMULATIONS_PER_MOVE = 400

# exploration parameters 
# TODO: change these values
C_base = 20000 # defines how CPUCT grows
C_init = 2

# if stochastic, use epsilon-greedy to sometimes pick moves from the prob dist
EPSILON = 0.2

# limit the amount of moves played in a game
MAX_PUZZLE_MOVES = 4
MAX_GAME_MOVES = 100

# ============= NEURAL NETWORK INPUTS =============
# 2 players, 6 pieces, 8x8 board
n = 8  # board size
# non boolean values: pieces for every player + the square for en passant
# boolean values: side to move, castling rights for every side and every player, is repitition
amount_of_input_planes =  (2*6 + 1) + (1 + 4 + 1)
INPUT_SHAPE = (n, n, amount_of_input_planes)

# ============= NEURAL NETWORK OUTPUTS =============
# the model has 2 outputs: policy and value
# ouput_shape[0] should be the number of possible moves
#       * 8x8 board: 8*8=64 possible actions
#       * 56 possible queen-like moves (horizontal/vertical/diagonal)
#       * 8 possible knight moves (every direction)
#       * 9 possible underpromotions
#   total values: 8*8*(56+8+9) = 4672
# ouput_shape[1] should be 1: a scalar value (v)
# 73 planes for chess:
queen_planes = 56
knight_planes = 8
underpromotion_planes = 9
amount_of_planes = queen_planes + knight_planes + underpromotion_planes
# the output shape for the vector
OUTPUT_SHAPE = (8*8*amount_of_planes, 1)


# ============= NEURAL NETWORK PARAMETERS =============
# TODO: change if necessary. AZ started with 0.2 and then dropped three times to 0.02, 0.002 and 0.0002
LEARNING_RATE = 0.002
# filters for the convolutional layers (AZ: 256)
CONVOLUTION_FILTERS = 256
# amount of hidden residual layers
# According to the AlphaGo Zero paper:
#    "For the larger run (40 block, 40 days), MCTS search parameters were re-optimised using the neural network trained in the smaller run (20 block, 3 days)."
# ==> First train a small NN, then optimize longer with a larger NN.
AMOUNT_OF_RESIDUAL_BLOCKS = 19

# where to save the model
MODEL_FOLDER = './models/'

# ============= TRAINING PARAMETERS =============
BATCH_SIZE = 128
LOSS_PLOTS_FOLDER="./plots"

# ============= MEMORY CONFIGURATION =============
MEMORY_DIR = "./memory"
MAX_REPLAY_MEMORY = 1000000