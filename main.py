#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 23:50:07 2023

@author: siipola
"""

import projectLib as chGif

import time
import chess
import chess.svg
import numpy as np
import random
import sys
import torch



board = chess.Board()
chess.svg.board(board, size=350)

gg_ = chGif.feature_representation(3)

hidden_layer_width = 50
""" Shallow is a neural network, shallow is a synonym for two layer, or one 
    hidden layer neural network. """
shallow = chGif.shallow(hidden_layer_width, 'relu', gg_.n_features)
shallow.build()
eval_ = chGif.evaluator(shallow.neural_net)


while True:
    if not gg_.test_if_legal_moves(board):
        break
    
    board = gg_.play_turn(board)
    print(board.turn) # side to move
    print(bool(board.castling_rights & chess.BB_A1))
    print(bool(board.castling_rights & chess.BB_H1))
    print(board.has_kingside_castling_rights(chess.WHITE))
    print(chess.BB_H1)
    gg_.display_board(board)
    print(board)
    print(board.piece_at(chess.A1))
#        print(board.R(chess.WHITE))
    print(board.board_fen())

    hash_map = gg_.from_fen_to_piece_positions(board.board_fen())
    
    print(hash_map)
    vec = gg_.feature_representation(board)
    print(vec.shape)
    vec = torch.from_numpy(vec)
    vec = torch.reshape(vec, (1,75))
    vec = vec.to(torch.float)
    print(vec.shape)
#    sys.exit()
    pos_value = eval_.evaluate(vec)
    print(vec)
    print(pos_value)
    print('ok')
    sys.exit()

print(board)

display_board(board)










































