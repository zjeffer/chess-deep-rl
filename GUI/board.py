"""
    Ahira Justice, ADEFOKUN
    justiceahira@gmail.com
"""


import os
from typing import Tuple
import pygame
from pygame.locals import *

from pygame.surface import Surface

from .pieces import Piece, PieceColor, PieceType
from .fenparser import FenParser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'images')


class Board:
    btile = pygame.image.load(os.path.join(IMAGE_DIR, 'btile.png'))
    wtile = pygame.image.load(os.path.join(IMAGE_DIR, 'wtile.png'))

    def __init__(self, colors: dict, BGCOLOR: tuple, DISPLAYSURF: Surface, width: int, height: int):
        self.colors = colors
        self.BGCOLOR = BGCOLOR
        self.DISPLAYSURF = DISPLAYSURF

        self.WINDOWWIDTH = width
        self.WINDOWHEIGHT = height

        self.pieceRect = []

        # create boardRect procedurally, based on the width and height of the screen
        square_size = min(self.WINDOWWIDTH, self.WINDOWHEIGHT) / 8
        print(square_size)

        squares = []
        for y in range(8):
            squares.append([])
            for x in range(8):
                squares[-1].append((x*square_size, y*square_size))

        Board.boardRect = squares
        Board.posb = (*squares[0], *squares[1])
        Board.posw = (*squares[-2], *squares[-1])

        self.square_size = square_size

        # change size of wtile and btile
        Board.wtile = pygame.transform.scale(self.wtile, (int(self.square_size), int(self.square_size)))
        Board.btile = pygame.transform.scale(self.btile, (int(self.square_size), int(self.square_size)))

        # for click events
        self.from_square = None
        self.to_square = None


    def get_click_events(self):
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                print(f"Mouse position: {x}, {y}")
                if self.from_square is None:
                    self.from_square = self.get_square_on_pos(x, y)
                elif self.get_square_on_pos(x, y) != self.from_square:
                    self.to_square = self.get_square_on_pos(x, y)
                    self.move_piece(self.from_square, self.to_square)
                    self.from_square = None
                    self.to_square = None

    def get_square_on_pos(self, x, y):
        # calculate the square based on mouse position
        square = (int(x / self.square_size), int(y / self.square_size))
        print(f"Square: {square} on position: {x}, {y}")
        return square
    
    def move_piece(self, from_square, to_square):
        print(f"Moving piece from {from_square} to {to_square}")
        

    def displayBoard(self):
        """
        Displays the board tiles on the screen
        """

        # fill the background with the background color
        self.DISPLAYSURF.fill(self.BGCOLOR)
        # draw a rectangle around the board
        pygame.draw.rect(self.DISPLAYSURF, self.colors['Black'], (0, 0, self.WINDOWWIDTH, self.WINDOWHEIGHT), 10)
        # draw the board tiles
        self.drawTiles()


    def drawTiles(self):
        for i in range(1, len(Board.boardRect)+1):
            for j in range(1, len(Board.boardRect[i-1])+1):
                if self.isOdd(i):
                    if self.isOdd(j):
                        self.DISPLAYSURF.blit(Board.wtile, Board.boardRect[i-1][j-1])
                    elif self.isEven(j):
                        self.DISPLAYSURF.blit(Board.btile, Board.boardRect[i-1][j-1])
                elif self.isEven(i):
                    if self.isOdd(j):
                        self.DISPLAYSURF.blit(Board.btile, Board.boardRect[i-1][j-1])
                    elif self.isEven(j):
                        self.DISPLAYSURF.blit(Board.wtile, Board.boardRect[i-1][j-1])


    def isOdd(self, number):
        if number % 2 == 1:
            return True


    def isEven(self, number):
        if number % 2 == 0:
            return True


    def drawPieces(self):
        self.mapPieces()

        for piece in self.pieceRect:
            piece.displayPiece()


    def mapPieces(self):
        for i in range(len(Board.posb)):
            if i in [0, 1, 2, 3, 4, 5, 6, 7]:
                piece = self.createPiece(PieceColor.BLACK, PieceType.PAWN, Board.posb[i])
                self.pieceRect.append(piece)
            elif i in [8, 15]:
                piece = self.createPiece(PieceColor.BLACK, PieceType.ROOK, Board.posb[i])
                self.pieceRect.append(piece)
            elif i in [9, 14]:
                piece = self.createPiece(PieceColor.BLACK, PieceType.KNIGHT, Board.posb[i])
                self.pieceRect.append(piece)
            elif i in [10, 13]:
                piece = self.createPiece(PieceColor.BLACK, PieceType.BISHOP, Board.posb[i])
                self.pieceRect.append(piece)
            elif i in [11]:
                piece = self.createPiece(PieceColor.BLACK, PieceType.QUEEN, Board.posb[i])
                self.pieceRect.append(piece)
            elif i in [12]:
                piece = self.createPiece(PieceColor.BLACK, PieceType.KING, Board.posb[i])
                self.pieceRect.append(piece)
        
        for i in range(len(Board.posw)):
            if i in [0, 1, 2, 3, 4, 5, 6, 7]:
                piece = self.createPiece(PieceColor.WHITE, PieceType.PAWN, Board.posw[i])
                self.pieceRect.append(piece)
            elif i in [8, 15]:
                piece = self.createPiece(PieceColor.WHITE, PieceType.ROOK, Board.posw[i])
                self.pieceRect.append(piece)
            elif i in [9, 14]:
                piece = self.createPiece(PieceColor.WHITE, PieceType.KNIGHT, Board.posw[i])
                self.pieceRect.append(piece)
            elif i in [10, 13]:
                piece = self.createPiece(PieceColor.WHITE, PieceType.BISHOP, Board.posw[i])
                self.pieceRect.append(piece)
            elif i in [11]:
                piece = self.createPiece(PieceColor.WHITE, PieceType.QUEEN, Board.posw[i])
                self.pieceRect.append(piece)
            elif i in [12]:
                piece = self.createPiece(PieceColor.WHITE, PieceType.KING, Board.posw[i])
                self.pieceRect.append(piece)


    def createPiece(self, color: PieceColor, piece_type: PieceType, position):
        piece = Piece(color, piece_type, self.DISPLAYSURF, self.square_size)
        piece.setPosition(position)
        return piece

    
    def updatePieces(self, fen):
        self.pieceRect: Piece = []
        fp = FenParser(fen)
        fenboard = fp.parse()

        for i in range(len(fenboard)):
            for j in range(len(fenboard[i])):
                if fenboard[i][j] in ['b', 'B']:
                    if fenboard[i][j] == 'b':
                        piece = self.createPiece(PieceColor.BLACK, PieceType.BISHOP, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                    elif fenboard[i][j] == 'B':
                        piece = self.createPiece(PieceColor.WHITE, PieceType.BISHOP, Board.boardRect[i][j])
                        self.pieceRect.append(piece)

                elif fenboard[i][j] in ['k', 'K']:
                    if fenboard[i][j] == 'k':
                        piece = self.createPiece(PieceColor.BLACK, PieceType.KING, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                    elif fenboard[i][j] == 'K':
                        piece = self.createPiece(PieceColor.WHITE, PieceType.KING, Board.boardRect[i][j])
                        self.pieceRect.append(piece)

                elif fenboard[i][j] in ['n', 'N']:
                    if fenboard[i][j] == 'n':
                        piece = self.createPiece(PieceColor.BLACK, PieceType.KNIGHT, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                    elif fenboard[i][j] == 'N':
                        piece = self.createPiece(PieceColor.WHITE, PieceType.KNIGHT, Board.boardRect[i][j])
                        self.pieceRect.append(piece)

                elif fenboard[i][j] in ['p', 'P']:
                    if fenboard[i][j] == 'p':
                        piece = self.createPiece(PieceColor.BLACK, PieceType.PAWN, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                    elif fenboard[i][j] == 'P':
                        piece = self.createPiece(PieceColor.WHITE, PieceType.PAWN, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                
                elif fenboard[i][j] in ['q', 'Q']:
                    if fenboard[i][j] == 'q':
                        piece = self.createPiece(PieceColor.BLACK, PieceType.QUEEN, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                    elif fenboard[i][j] == 'Q':
                        piece = self.createPiece(PieceColor.WHITE, PieceType.QUEEN, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                
                elif fenboard[i][j] in ['r', 'R']:
                    if fenboard[i][j] == 'r':
                        piece = self.createPiece(PieceColor.BLACK, PieceType.ROOK, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                    elif fenboard[i][j] == 'R':
                        piece = self.createPiece(PieceColor.WHITE, PieceType.ROOK, Board.boardRect[i][j])
                        self.pieceRect.append(piece)
                        
        for piece in self.pieceRect:
            piece.displayPiece()

