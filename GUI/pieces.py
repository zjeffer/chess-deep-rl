"""
    Ahira Justice, ADEFOKUN
    justiceahira@gmail.com
"""


import os
import pygame


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'images')


class PieceColor:
    BLACK = 'BLACK'
    WHITE = 'WHITE'

class PieceType:
    BISHOP = 'BISHOP'
    KING = 'KING'
    KNIGHT = 'KNIGHT'
    PAWN = 'PAWN'
    QUEEN = 'QUEEN'
    ROOK = 'ROOK'


class Piece:
    bBishop = pygame.image.load(os.path.join(IMAGE_DIR, 'bB.png'))
    bKing = pygame.image.load(os.path.join(IMAGE_DIR, 'bK.png'))
    bKnight = pygame.image.load(os.path.join(IMAGE_DIR, 'bN.png'))
    bPawn = pygame.image.load(os.path.join(IMAGE_DIR, 'bP.png'))
    bQueen = pygame.image.load(os.path.join(IMAGE_DIR, 'bQ.png'))
    bRook = pygame.image.load(os.path.join(IMAGE_DIR, 'bR.png'))

    wBishop = pygame.image.load(os.path.join(IMAGE_DIR, 'wB.png'))
    wKing = pygame.image.load(os.path.join(IMAGE_DIR, 'wK.png'))
    wKnight = pygame.image.load(os.path.join(IMAGE_DIR, 'wN.png'))
    wPawn = pygame.image.load(os.path.join(IMAGE_DIR, 'wP.png'))
    wQueen = pygame.image.load(os.path.join(IMAGE_DIR, 'wQ.png'))
    wRook = pygame.image.load(os.path.join(IMAGE_DIR, 'wR.png'))

    def __init__(self, color, piece, DISPLAYSURF, size):
        self.position = None
        self.sprite = None
        self.DISPLAYSURF = DISPLAYSURF
        self.size = size

        self.color = color
        self.piece = piece

        self.setSprite()

    def setPosition(self, position):
        self.position = position
    

    def setSprite(self):        
        if self.piece == PieceType.BISHOP:
            if self.color == PieceColor.BLACK:
                self.sprite = Piece.bBishop
            elif self.color == PieceColor.WHITE:
                self.sprite = Piece.wBishop
        
        elif self.piece == PieceType.KING:
            if self.color == PieceColor.BLACK:
                self.sprite = Piece.bKing
            elif self.color == PieceColor.WHITE:
                self.sprite = Piece.wKing
        
        elif self.piece == PieceType.KNIGHT:
            if self.color == PieceColor.BLACK:
                self.sprite = Piece.bKnight
            if self.color == PieceColor.WHITE:
                self.sprite = Piece.wKnight
        
        elif self.piece == PieceType.PAWN:
            if self.color == PieceColor.BLACK:
                self.sprite = Piece.bPawn
            elif self.color == PieceColor.WHITE:
                self.sprite = Piece.wPawn
        
        elif self.piece == PieceType.QUEEN:
            if self.color == PieceColor.BLACK:
                self.sprite = Piece.bQueen
            elif self.color == PieceColor.WHITE:
                self.sprite = Piece.wQueen
        
        elif self.piece == PieceType.ROOK:
            if self.color == PieceColor.BLACK:
                self.sprite = Piece.bRook
            elif self.color == PieceColor.WHITE:
                self.sprite = Piece.wRook


    def displayPiece(self):
        self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))
        self.DISPLAYSURF.blit(self.sprite, self.position)
