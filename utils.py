import chess
from chess import Move, PieceType
import numpy as np
from PIL import Image
import time
from mapper import Mapping


def save_input_state_to_imgs(input_state: np.ndarray, path: str, names: list = None, only_full: bool = False):
    """
    Save the input state to images
    """
    start_time = time.time()
    if not only_full:
        for index, plane in enumerate(input_state):
            # save boolean 2d array to image
            img = Image.fromarray(plane)
            # save image
            if names is not None and len(names) == len(input_state):
                # print index, with one leading 0
                img.save(f"{path}/{index:02d}-{names[index]}.png")
            else:
                img.save(f"{path}/{index:02d}.png")

    # full image of all states
    # convert booleans to integers
    input_state = np.array(input_state)*np.uint8(255)
    # pad input_state with grey values
    input_state = np.pad(input_state, ((0, 0), (1, 1), (1, 1)),
                         'constant', constant_values=128)

    full_array = np.concatenate(input_state, axis=1)
    # more padding
    full_array = np.pad(full_array, ((4, 4), (5, 5)),
                        'constant', constant_values=128)
    img = Image.fromarray(full_array)
    img.save(f"{path}/full.png")
    print(
        f"*** Saving to images: {(time.time() - start_time):.6f} seconds ***")


def save_output_state_to_imgs(output_state: np.ndarray, path: str, name: str = "full"):
    """
    Save the output state to images
    """
    start_time = time.time()
    # full image of all states
    # pad input_state with grey values
    output_state = np.pad(output_state.astype(float)*255, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=128)
    full_array = np.concatenate(output_state, axis=1)
    # more padding
    full_array = np.pad(full_array, ((4, 4), (5, 5)), 'constant', constant_values=128)
    img = Image.fromarray(full_array.astype(np.uint8))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(f"{path}/{name}.png")
    print(
        f"*** Saving to images: {(time.time() - start_time):.6f} seconds ***")

# timer function decorator
def timer_function(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def moves_to_output_vector(moves: dict, board: chess.Board) -> np.ndarray:
    """
    Convert a dictionary of moves to a vector of probabilities
    """
    vector = np.zeros((73, 8, 8), dtype=np.float32)
    for move in moves:
        plane_index, row, col = move_to_plane_index(move, board)
        vector[plane_index, row, col] = moves[move]
    return np.asarray(vector)
    

def move_to_plane_index(move: str, board: chess.Board):
    # convert move to plane index
    move: Move = Move.from_uci(move)
    # get start and end position
    from_square = move.from_square
    to_square = move.to_square
    # get piece
    piece: chess.Piece = board.piece_at(from_square)

    if piece is None:
            raise Exception(f"No piece at {from_square}")

    plane_index: int = None

    if move.promotion and move.promotion != chess.QUEEN:
        piece_type, direction = Mapping.get_underpromotion_move(
            move.promotion, from_square, to_square
        )
        plane_index = Mapping.mapper[piece_type][1 - direction]
    else:
        if piece.piece_type == chess.KNIGHT:
            # get direction
                direction = Mapping.get_knight_move(from_square, to_square)
                plane_index = Mapping.mapper[direction]
        else:
            # get direction of queen-type move
            direction, distance = Mapping.get_queenlike_move(
                from_square, to_square)
            plane_index = Mapping.mapper[direction][np.abs(distance)-1]
    row = from_square % 8
    col = 7 - (from_square // 8)
    return (plane_index, row, col)