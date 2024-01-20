#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:45:08 2022

@author: siipola
"""

import chess
board = chess.Board()
# get the current square index of the white king
king_square_index = board.king(chess.WHITE)
queen_square_index = board.queen(chess.WHITE)
print(king_square_index)