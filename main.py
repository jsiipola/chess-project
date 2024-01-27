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
import time

""" setting device on GPU if available, else CPU """
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dev = torch.device('cpu')
if (torch.cuda.is_available()):
    torch.cuda.empty_cache()
    # sys.exit()
# dev = 'cpu'
print('Using device:', dev)
print()
device = torch.device(dev)


board = chess.Board()
board2 = board.copy()
print(board)
print(board2)
chess.svg.board(board, size=350)

gg_ = chGif.feature_representation(3)

hidden_layer_width = 50
""" Shallow is a neural network, shallow is a synonym for two layer, or one 
    hidden layer neural network. """
shallow = chGif.shallow(hidden_layer_width, 'tanh', gg_.n_features)
shallow.build()
eval_ = chGif.evaluator(shallow.neural_net)

""" Initialize the matchstate which controls the game. """
gameMaster = chGif.matchState('AI', 'AI', eval_, gg_, 2, device)
""" Start playing the chess game. """
gameMaster.play()
sys.exit()
treeResearcher = chGif.DepthSearcher(evaluationFunction = eval_, featureRep = gg_, device = device)
pos_value, state, move = treeResearcher.minimax(position = board, depth = 2, maximizingPlayer = True)
print(move)
# t0 = time.time()
# pos_value = treeResearcher.minimax_alphaBetaPruning(position = board, depth = 5,
                                                    # maximizingPlayer = True)
# t1 = time.time()
# print(pos_value)
# print(t1-t0)



sys.exit()
while True:
    if not gg_.test_if_legal_moves(board):
        break
    
    board = gg_.play_turn(board)
    
    # board2 = copy(board)
    # sys.exit()
    print(board.turn) # side to move
    print(bool(board.castling_rights & chess.BB_A1))
    # print(board)
    # sys.exit()
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
    """ feature_representation return the situation of the board as a vector. """
    vec = gg_.feature_representation(board)
    print(vec.shape)
    print()
    # vec = torch.from_numpy(vec)
    # vec = torch.reshape(vec, (1,75))
    # vec = vec.to(torch.float)
    print(vec.shape)
#    sys.exit()
    pos_value = eval_.evaluate(vec)
    print(vec)
    print(pos_value)
    print('ok')
    
    print(board)
    print(board2)
    
    sys.exit()


display_board(board)









































