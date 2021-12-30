# config file: includes parameters for the model and the mcts tree

# ============= MCTS =============
AMOUNT_OF_SIMULATIONS = 200
MAX_DEPTH = 100

# ============= NEURAL NETWORK INPUTS =============
# 2 players, 6 pieces, 8x8 board
n = 8  # board size
t = 8  # amount of timesteps
m = 2 * 6 + 1  # pieces for every player + the square for en passant
# additional values: move counter, repitition counter, side to move, castling rights for every side and every player
l = 1 + 1 + 1 + (2*2)
INPUT_SHAPE = (n, n, m*t+l)

# ============= NEURAL NETWORK OUTPUTS =============
# the model has 2 outputs: policy and value
# ouput_shape[0] should be the number of possible moves
#       * 8x8 board: 8*8=64 possible actions
#       * 56 possible queen-like moves (horizontal/vertical/diagonal)
#       * 8 possible knight moves (every direction)
#       * 9 possible underpromotions
#   total values: 8*8*(56+8+9) = 4672
# ouput_shape[1] should be 1: a scalar value (v)
OUTPUT_SHAPE = (8*8*(56+8+9), 1)

AMOUNT_OF_RESIDUAL_BLOCKS = 19
