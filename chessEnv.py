import chess


# class for chess environment
class ChessEnv:
    def __init__(self):
        """
        Initialize the chess environment
        """
        # the chessboard
        self.board = chess.Board()

    def reset(self):
        """
        Reset everything
        """
        self.board = chess.Board()
    
    @staticmethod
    def board_to_state(board):
        """
        Convert board to a state that is interpretable by the agent
        """
        # get fen string
        fen = board.fen()
        # get pieces
        pieces = fen.split(" ")[0]
        # TODO: figure out in what format the state should be according to the model
        return fen
 

    def step(self, action):
        """ 
        Perform an action on the board and return the reward
        """
        pass

    

if __name__ == "__main__":
    chessEnv = ChessEnv()
    print(ChessEnv.board_to_state(chessEnv.board))