import time
from agent import Agent
from chessEnv import ChessEnv
import utils
import logging
logging.basicConfig(level=logging.INFO)


def timer_function(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


class Test:
    def __init__(self):
        self.reset()

    def reset(self):
        # don't build the model (it's not needed for these tests)
        white = Agent(build_model=False)
        black = Agent(build_model=False)
        self.env = ChessEnv(white, black)

    @timer_function
    def run_tests(self, n: int = 1):

        for _ in range(n):

            self.reset()
            # test en passant
            self.env.board.push_san("e4")
            self.env.board.push_san("a5")
            self.env.board.push_san("e5")
            self.env.board.push_san("Nc6")
            self.env.board.push_san("d4")
            self.env.board.push_san("d5")

            # test input_state
            input_state = self.env.board_to_state(self.env.board)

        # names = ['white_turn', 'black_turn',
        # 	'castling_king_white', 'castling_king_black', 'castling_queen_white', 'castling_queen_black',
        # 	'is_repitition',
        # 	'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens', 'white_king',
        # 	'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens', 'black_king',
        # 	'en_passant']
        # utils.save_input_state_to_imgs(input_state, 'tests/input_planes', names)
