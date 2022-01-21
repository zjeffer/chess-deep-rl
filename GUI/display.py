"""
    Ahira Justice, ADEFOKUN
    justiceahira@gmail.com
"""


import os
import sys
import pygame
from pygame.locals import *

from .board import Board

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
    def __init__(self, width: int, height: int, fen: str=''):
        self.gameboard = None
        self.WINDOWWIDTH = width
        self.WINDOWHEIGHT = height

        self.start(fen)

    def start(self, fen='') -> None:
        pygame.init()

        # Setting up the GUI window.
        self.DISPLAYSURF = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))
        pygame.display.set_caption('LOCI')
        BASICFONT = pygame.font.SysFont('roboto', BASICFONTSIZE)

        self.checkForQuit()

        self.DISPLAYSURF.fill(BGCOLOR)
        self.gameboard = Board(colors, BGCOLOR, self.DISPLAYSURF, self.WINDOWWIDTH, self.WINDOWHEIGHT)
        self.gameboard.displayBoard()

        if (fen):
            self.gameboard.updatePieces(fen)
        else:
            self.gameboard.drawPieces()
        
        self.draw()

    def update(self, fen):
        self.checkForQuit()
        self.gameboard.displayBoard()
        self.gameboard.updatePieces(fen)

        self.draw()

    def draw(self):
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        self.get_events()

    def get_events(self):
        self.checkForQuit()
        self.gameboard.get_click_events()


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


    
