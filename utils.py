from typing import Iterator
from chess import Move
import numpy as np
from PIL import Image
import time

import threading
import queue


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

class Worker(threading.Thread):
    def __init__(self, moves: Iterator[Move], *args, **kwargs):
        self.moves = moves
        super(Worker, self).__init__(*args, **kwargs)
    
    def run(self):
        while True:
            try: 
                move = next(self.moves)
            except StopIteration:
                print("End of moves")
                return
            return move

# q = queue.Queue()
# for move in [1, 2, 3, 4]:
#     q.put_nowait(move)
# for _ in range(20):
#     Worker(q).start()
# q.join()