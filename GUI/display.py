"""
    Ahira Justice, ADEFOKUN
    justiceahira@gmail.com
"""


import os
import sys
import pygame
from pygame.locals import *

from .board import Board

import chess

os.environ['SDL_VIDEO_CENTERED'] = '1' # Centre display window.

FPS = 30
FPSCLOCK = pygame.time.Clock()

colors = {
    'Ash':  ( 75,  75,  75),
    'White':(255, 255, 255),
    'Black':(  0,   0,   0),
}

BGCOLOR = colors['Ash']

BASICFONTSIZE = 30

class GUI:
    def __init__(self, width: int, height: int, player: bool, fen: str=chess.STARTING_FEN):
        self.gameboard = None
        self.WINDOWWIDTH = width
        self.WINDOWHEIGHT = height
        self.player = player

        self.fen = fen

        self.start()

    def start(self) -> None:
        pygame.init()

        # Setting up the GUI window.
        self.DISPLAYSURF = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))
        pygame.display.set_caption('LOCI')
        BASICFONT = pygame.font.SysFont('roboto', BASICFONTSIZE)

        self.checkForQuit()

        self.DISPLAYSURF.fill(BGCOLOR)
        self.gameboard = Board(colors, BGCOLOR, self.DISPLAYSURF, self.WINDOWWIDTH, self.WINDOWHEIGHT)
        self.gameboard.displayBoard()
        
        self.draw()

    def make_move(self, move: chess.Move):
        self.gameboard.board.push(move)

    def draw(self):
        self.gameboard.displayBoard()
        self.gameboard.updatePieces()

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
                x, y = pygame.mouse.get_pos()
                if event.button == 3:
                    # right mouse button => clear 
                    self.gameboard.from_square = None
                    self.gameboard.to_square = None
                elif self.gameboard.from_square is None:
                    # fist click: get from_square
                    self.gameboard.from_square = self.gameboard.get_square_on_pos(x, y)
                elif self.gameboard.get_square_on_pos(x, y) != self.gameboard.from_square:
                    # from and to_square are different, try to move piece
                    self.gameboard.to_square = self.gameboard.get_square_on_pos(x, y)
                    self.gameboard.move_piece(self.gameboard.from_square, self.gameboard.to_square)
                    self.gameboard.from_square = None
                    self.gameboard.to_square = None

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


    
