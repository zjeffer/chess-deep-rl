"""
    Ahira Justice, ADEFOKUN
    justiceahira@gmail.com
"""


import os
import sys
import time
import pygame
from pygame.locals import *

import pygamepopup
from pygamepopup.menu_manager import MenuManager, InfoBox
from pygamepopup.components import Button

from .board import Board

import chess

os.environ['SDL_VIDEO_CENTERED'] = '1' # Centre display window.

FPS = 30
FPSCLOCK = pygame.time.Clock()

colors = {
    'Background':  ( 75,  75,  75),
    'White':(255, 255, 255),
    'Black':( 0,  0,  0),
}

BGCOLOR = colors['Background']

BASICFONTSIZE = 30

class GUI:
    def __init__(self, width: int, height: int, player: bool, fen: str=chess.STARTING_FEN):
        self.gameboard = None
        self.WINDOWWIDTH = width
        self.WINDOWHEIGHT = height
        self.player = player

        self.fen = fen

        # for moving pieces
        self.from_square = None
        self.to_square = None
        # for promotion
        self.promoting = False

        self.start()

    def start(self) -> None:
        pygame.init()
        pygame.display.set_caption('Chess with Reinforcement Learning')
        pygamepopup.init()

        # Setting up the GUI window.
        self.DISPLAYSURF = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))

        # set up popup manager
        self.menu_manager = MenuManager(screen=self.DISPLAYSURF)

        # BASICFONT = pygame.font.SysFont('roboto', BASICFONTSIZE)

        self.checkForQuit()

        self.DISPLAYSURF.fill(BGCOLOR)
        self.gameboard = Board(colors, BGCOLOR, self.DISPLAYSURF, self.WINDOWWIDTH, self.WINDOWHEIGHT)
        self.gameboard.displayBoard()

        self.promotion_menu = InfoBox(
            "Choose a piece to promote to:",
            [
                [
                    Button(title="Queen", callback=lambda: self.promote(chess.QUEEN), size=(100, 50)),
                ],
                [
                    Button(title="Rook", callback=lambda: self.promote(chess.ROOK), size=(100, 50)),
                ],
                [
                    Button(title="Bishop", callback=lambda: self.promote(chess.BISHOP), size=(100, 50)),
                ],
                [
                    Button(title="Knight", callback=lambda: self.promote(chess.KNIGHT), size=(100, 50)),
                ],
            ],
            has_close_button=False
        )
        
        self.draw()

    def promote(self, piece: chess.PieceType):
        print("Trying to promote piece")
        self.move_piece(piece)

    def show_promotion_menu(self):
        self.promoting = True
        self.menu_manager.open_menu(self.promotion_menu)

    def make_move(self, move: chess.Move):
        self.gameboard.board.push(move)

    def draw(self):
        self.gameboard.displayBoard()
        self.gameboard.updatePieces()

        if self.promoting:
            self.menu_manager.display()
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        self.get_events()

    def get_events(self):
        self.checkForQuit()
        self.get_click_events()

    def get_click_events(self):
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                # only move piece if the player is the current player
                if self.gameboard.board.turn != self.player:
                    print("Not your turn!")
                    continue
                if self.promoting:
                    self.menu_manager.click(event.button, event.pos)
                    continue
                x, y = pygame.mouse.get_pos()
                if event.button == 3:
                    # right mouse button => clear 
                    self.from_square = None
                    self.to_square = None
                elif self.from_square is None:
                    # fist click: get from_square
                    self.from_square = self.gameboard.get_square_on_pos(x, y)
                    print(f"from_square: {self.from_square}")
                elif self.gameboard.get_square_on_pos(x, y) != self.from_square:
                    # from and to_square are different, try to move piece
                    self.to_square = self.gameboard.get_square_on_pos(x, y)
                    print(f"to_square: {self.to_square}")
                    piece = self.gameboard.get_piece_to_move(self.from_square, self.to_square)
                    if piece is None:
                        print("Piece is none")
                        self.from_square = None
                        self.to_square = None
                        continue
                    if piece.color != self.gameboard.board.turn:
                        print("Wrong color")
                        self.from_square = None
                        self.to_square = None
                        continue
                    
                    to_square = Board.square_to_string(Board.tuple_to_square(*self.to_square))
                    if piece.piece_type == chess.PAWN and (to_square[1] == '8' or to_square[1] == '1'):
                        print("Promotion")
                        # get promotion from menu
                        self.promotion_choice: chess.PieceType = None
                        self.show_promotion_menu()
                    else:
                        self.move_piece(piece)


    def move_piece(self, piece: chess.PieceType):
        # move piece to to_square
        # create san from move
        try:
            from_square = Board.square_to_string(Board.tuple_to_square(*self.from_square))
            to_square = Board.square_to_string(Board.tuple_to_square(*self.to_square))
            move = chess.Move.from_uci(f"{from_square}{to_square}")
            if self.promoting:
                move.promotion = piece
                self.promoting = False
            if self.gameboard.board.is_legal(move):
                self.gameboard.board.push(move)
        except ValueError as e:
            print("Invalid move")
            raise(e)
        self.from_square = None
        self.to_square = None

    def terminate(self):
        pygame.quit()
        sys.exit()

    def checkForQuit(self):
        for event in pygame.event.get(QUIT): # get all the QUIT events
            self.terminate() #terminate if any QUIT events are present
        for event in pygame.event.get(KEYUP): # get all the KEYUP events
            if event.key == K_ESCAPE:
                self.terminate() # terminate if the KEYUP event was for the Esc key
            pygame.event.post(event) # put the other KEYUP event objects back
        
        return False


    
