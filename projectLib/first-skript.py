#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 08:43:08 2022

@author: siipola
"""

#import chess
#
#board = chess.Board()
#print(board)

import chess.svg
board = chess.Board()
chess.svg.board(board, size=350)





import chess
import chess.svg
import time

#print(chess.svg.piece(chess.Piece.from_symbol("R"))  )



import chess
import chess.svg

chess.svg.board(board = board, size = 300)  

## Setup board here 
display(board)
#time.sleep(0.5)
# moving players
board.push_san("e4")
display(board)
#time.sleep(0.5)
#chess.Move(chess.E7, chess.E5, "Q", None)
#board.Move("e7","e8","q")
move_ = chess.Move.from_uci("e7e5")
board.push(move_)
display(board)
#time.sleep(0.5)
move_ = chess.Move.from_uci("d2d4")
board.push(move_)
display(board)
#time.sleep(0.5)
move_ = chess.Move.from_uci("e5d4")
board.push(move_)
display(board)
#time.sleep(0.5)
print(board.turn)
move_ = chess.Move.from_uci("c2c3")
board.push(move_)
display(board)
#time.sleep(0.5)
print(board.turn)

move_ = chess.Move.from_uci("f1g2")
board.push(move_)
print('Is the board valid?')
print(board.is_valid())
print(chess.PAWN)
print(chess.KNIGHT)
#print(chess.knight) Does not work like this.
print(chess.piece_symbol(2))
print(chess.piece_name(2))
print(chess.A1)
print(chess.G8)
display(board)

print(chess.square_name(63) )

print(chess.square_file(12))

print(chess.square_distance(0, 10))

print(bool(board.castling_rights & chess.BB_H1))

print(board.fullmove_number)

display(board)
move_ = chess.Move.from_uci("c1c4")
print(move_ in board.legal_moves)
board.push(move_)
display(board)
print(board.legal_moves.count())
"""
board.turn

    board.turn gives information about the turn to move,
if return of this attribute is True then it is white's
turn, if the return is False then it is black's turn
to move.

board.push

    board.push makes the move to happen. One must give a move
for this function as an argument.

display

    display(board) plots an image of the chess board.

chess.Move.from_uci()

    With a command like chess.Move.from_uci("e7e5")
we can define a move of a piece. This can be given as an
argument for the board.push() function.
"""
#board.is_valid(move_)
#print(move_)




