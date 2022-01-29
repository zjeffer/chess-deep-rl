"""
    Ahira Justice, ADEFOKUN
    justiceahira@gmail.com
"""


import os
from typing import Tuple
import pygame
import chess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'images')


class PieceImage:
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

    pieceImages = [
        (wPawn, bPawn),
        (wKnight, bKnight),
        (wBishop, bBishop),
        (wRook, bRook),
        (wQueen, bQueen),
        (wKing, bKing)
    ]

    def __init__(self, color: chess.Color, piece_type: chess.PieceType, DISPLAYSURF, size: int):
        self.position = None
        self.sprite = None
        self.DISPLAYSURF = DISPLAYSURF
        self.size = size

        self.color = color
        self.piece_type = piece_type

        self.setSprite()

    def setPosition(self, position: Tuple[int, int]):
        self.position = position
    

    def setSprite(self):
        self.sprite = self.pieceImages[self.piece_type - 1][int(not self.color)]


    def displayPiece(self):
        self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))
        self.DISPLAYSURF.blit(self.sprite, self.position)
