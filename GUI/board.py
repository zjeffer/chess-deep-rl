"""
    Ahira Justice, ADEFOKUN
    justiceahira@gmail.com
"""


import os
from typing import Tuple
import pygame
from pygame.locals import *

import chess

from pygame.surface import Surface

from .pieces import PieceImage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'images')


class Board:
    btile = pygame.image.load(os.path.join(IMAGE_DIR, 'btile.png'))
    wtile = pygame.image.load(os.path.join(IMAGE_DIR, 'wtile.png'))
    tile_selected = pygame.image.load(os.path.join(IMAGE_DIR, 'tile-selected.png'))

    def __init__(self, colors: dict, BGCOLOR: tuple, DISPLAYSURF: Surface, width: int, height: int):
        self.colors = colors
        self.BGCOLOR = BGCOLOR
        self.DISPLAYSURF = DISPLAYSURF

        self.WINDOWWIDTH = width
        self.WINDOWHEIGHT = height

        self.pieceRect = []

        # create boardRect procedurally, based on the width and height of the screen
        square_size = min(self.WINDOWWIDTH, self.WINDOWHEIGHT) / 8

        squares = []
        for y in range(8):
            squares.append([])
            for x in range(8):
                squares[-1].append((x*square_size, y*square_size))

        Board.boardRect = squares

        self.square_size = square_size

        # change size of wtile and btile
        Board.wtile = pygame.transform.scale(
            Board.wtile, (int(self.square_size), int(self.square_size)))
        Board.btile = pygame.transform.scale(
            Board.btile, (int(self.square_size), int(self.square_size)))
        Board.tile_selected = pygame.transform.scale(
            Board.tile_selected, (int(self.square_size), int(self.square_size)))
        # keep the currently selected square
        self.selected_square = None


        self.board = chess.Board()
    

    def get_square_on_pos(self, x, y) -> Tuple[int, int]:
        # calculate the square based on mouse position
        square = (int(x / self.square_size), int(y / self.square_size))
        return square

    @staticmethod
    def square_to_tuple(square: int) -> Tuple[int, int]:
        return (square % 8, 7 - (square // 8))
    
    @staticmethod
    def tuple_to_square(x: int, y: int) -> int:
        return (7 - y)*8 + x

    @staticmethod
    def square_to_string(square: int) -> str:
        return chess.square_name(square)

    def get_piece_to_move(self, from_square, to_square) -> chess.Piece:
        from_square = Board.tuple_to_square(*from_square)
        to_square = Board.tuple_to_square(*to_square)

        # get piece from from_square
        piece = self.board.piece_at(from_square)
        
        return piece
        

    def displayBoard(self) -> None:
        """
        Displays the board tiles on the screen
        """
        # fill the background with the background color
        self.DISPLAYSURF.fill(self.BGCOLOR)
        # draw a rectangle around the board
        pygame.draw.rect(
            self.DISPLAYSURF, self.colors['Black'], (0, 0, self.WINDOWWIDTH, self.WINDOWHEIGHT), 10)
        # draw the board tiles
        self.drawTiles()

    def drawTiles(self):
        for i in range(len(Board.boardRect)):
            for j in range(len(Board.boardRect[i])):
                if self.is_selected(i, j):
                    tile = Board.tile_selected
                elif Board.isEven(i) and Board.isEven(j) or not Board.isEven(i) and not Board.isEven(j):
                        tile = Board.wtile
                else:
                        tile = Board.btile
                self.DISPLAYSURF.blit(tile, Board.boardRect[i][j])

    def is_selected(self, i: int, j: int) -> bool:
        """
        Returns True if a tile is selected, else false
        """
        return self.selected_square == (j, i)

    @staticmethod
    def isEven(n):
        return n % 2 == 0

    def createPiece(self, color: chess.Color, piece_type: chess.PieceType, square: int):
        piece = PieceImage(color, piece_type,
                           self.DISPLAYSURF, self.square_size)
        # convert square to pixel position
        position = tuple(self.square_size*x for x in square)
        piece.setPosition(position)
        return piece

    def updatePieces(self):
        # get pieces from fen
        pieces = self.board.piece_map()
        self.pieceRect: list[PieceImage] = []

        for square in pieces:
            piece: chess.Piece = pieces[square]
            square = Board.square_to_tuple(square)
            piece_image = self.createPiece(
                piece.color, piece.piece_type, square)
            self.pieceRect.append(piece_image)

        for piece_image in self.pieceRect:
            piece_image.displayPiece()
