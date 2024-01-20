#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 08:22:22 2022

@author: siipola
"""

import time
import chess
import chess.svg
import numpy as np
import random

"""
Computer plays random moves.

"""

def display_board(board):
    display(board)
    time.sleep(1)

def play_turn(board):
    move_list = []
    for move in board.legal_moves:
        move_list.append(move)
#    move_list = random.shuffle(move_list)
    k = board.legal_moves.count()
    x = np.random.rand()
    for i in range(k):
        i += 1
        if (float(i)/float(k)) > x:
            break
    move_index = i-1
    # after break i is the correct index.
#    print(move_list)
    move = move_list[move_index]
    board.push(move)
    return board

def test_if_legal_moves(board):
    g = board.legal_moves.count()
    if g>0:
        return True
    else:
        return False
    
def main():
    board = chess.Board()
    chess.svg.board(board, size=350)
    
    while True:
        if not test_if_legal_moves(board):
            break
        
        board = play_turn(board)
        display_board(board)

    display_board(board)


if __name__ == "__main__":
    main()











