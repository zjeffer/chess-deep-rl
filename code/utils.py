import socket
import chess
from chess import Move, PieceType
import numpy as np
from PIL import Image
import time
from mapper import Mapping
import config
from node import Node

def save_input_state_to_imgs(input_state: np.ndarray, path: str):
    """
    Save an input state to images
    """
    start_time = time.time()
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
    Save an output state to images
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

def time_function(func):
    """
    Decorator to time a function
    """
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
    """"
    Convert a move to a plane index and the row and column on the board
    """
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

def recvall(sock: socket.socket, count: int = 0) -> bytes:
    """
    Function to continuously receive data from a socket
    """
    buffer = b''
    if count == 0:
        while True:
            part = sock.recv(config.SOCKET_BUFFER_SIZE)
            buffer += part
            if len(part) < config.SOCKET_BUFFER_SIZE:
                break
    else:
        while count > 0:
            part = sock.recv(config.SOCKET_BUFFER_SIZE)
            buffer += part
            count -= len(part)
    return buffer
    
def get_height_of_tree(node: Node):
    if node is None:
        return 0

    h = 0
    for edge in node.edges:
        h = max(h, get_height_of_tree(edge.output_node))
    return h + 1

if __name__ == "__main__":
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Bc5")
    board.push_san("Nc3")
    board.push_san("Nf6")
    board.push_san("Bb5")

    from chessEnv import ChessEnv

    from agent import Agent
    agent = Agent(local_predictions=True, model_path="models/model-2022-04-13_19:39:50.h5", state=board.fen())

    input_state = ChessEnv.state_to_input(board.fen())
    p, v = agent.predict(input_state)

    print(p)

    probabilities = p.reshape(config.amount_of_planes, config.n, config.n)

    # input_state = np.reshape(input_state, (19, 8, 8))
    # save_input_state_to_imgs(input_state, "tests")
    save_output_state_to_imgs(probabilities, "./", "output_state")
