#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 22:41:51 2024

@author: jsiipola
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import chess
import chess.svg
import numpy as np
import random
import sys
import torch
import os 
from .depth_search import DepthSearcher
# from . import depth_searcher  # explicit relative import

from .feature_representation import feature_representation


class matchState(object):
    """ Class to somehow controll the game. """
    def __init__(self, white, black, evaluationFunction, featureRep, 
                 searchDepth, device):
        self.device = device
        if white == 'AI':
            self.playerWhite = AIchessAgent('white', evaluationFunction, featureRep, 
                                             searchDepth, device)
        else:
            self.playerWhite = 'human'
        if black == 'AI':
            self.playerBlack = AIchessAgent('black', evaluationFunction, featureRep,
                                            searchDepth, device)
        else:
            self.playerBlack = 'human'
        self.board = board = chess.Board()
        # chess.svg.board(board, size=350)
        self.moves = [] # List of oves that have been made
        self.movesNum = 0 # Number of moves that have been made.
        self.gameOn = False
        self.winner = None
        self.whoToMove = 0 # 0 white, 1 black
        self.result = None

    def whichPlayerIsToMove(self):
        return self.whoToMove

    def play_a_Turn(self, turn_):
            if turn_ == 0:
                move = self.playerWhite.playTurn(self.board)
            else:
                move = self.playerBlack.playTurn(self.board)
            self.board.push(move)


    def detect_draws(self):
        """ Return a True if the game has run into draw. Otherwise returns a False. """
        is_draw = False
        reason = None
        if self.board.is_stalemate():
            is_draw = True
            reason = 'stalemate'
        elif self.board.is_insufficient_material():
            is_draw = True
            reason = 'insufficient_material'
        elif self.board.can_claim_threefold_repetition():
            is_draw = True
            reason = 'repetition'
#        if board.halfmove_clock()
        elif self.board.can_claim_fifty_moves():
            is_draw = True
            reason = 'fifty moves'
        elif self.board.can_claim_draw():
            is_draw = True
            reason = 'unknown'
        # elif self.board.is_fivefold_repetition():
            # is_draw = True
        # elif self.board.is_seventyfive_moves():
            # is_draw = True
        return is_draw, reason
    
    def setTheEndResults(self, reason):
        self.result = reason
        if self.result == 'checkmate':
            self.winner = self.whoToMove
        else:
            self.winner = None
        
    
    def detect_check_mate(self):
        return self.board.is_checkmate()

    def isGameOver(self):
        """ A method to check if the has ended. Game may end to a checkmate or
            to a draw. """
        if self.detect_check_mate() == True:
            return True, 'checkmate'
        isDraw, reason = self.detect_draws()
        if isDraw:
            return isDraw, reason
        return False, None

    def play(self):
        self.gameOn = True
        moveCount = 0
        while(self.gameOn):
            turn_ = self.whichPlayerIsToMove()
            self.play_a_Turn(turn_)
            print(self.board)
            
            outcome, reason = self.isGameOver()
            if outcome == True:
                self.gameOn = False
                self.setTheEndResults(reason)
            moveCount += 1
            self.whoToMove = moveCount % 2
            print(str(moveCount) + ' moves being made.')
            self.display_board()
            time.sleep(0.3)
        print('Game ended.')
        print('Result: ' + str(self.result))
        if self.result == 'checkmate':
            if self.winner == 0:
                print('White wins!')
            else:
                print('Black wins!')
            
    def display_board(self):
        display(self.board)
        # time.sleep(1)


class AIchessAgent(object):
    def __init__(self, color, evaluationFunction, featureRep, searchDepth,
                 device):
        self.color = color
        self.device = device
        self.feature_representation = featureRep
        self.treeSearch = DepthSearcher(evaluationFunction, featureRep, device)
        self.depth = 2
        if color == 0 or color == 'white':
            self.maximizingPlayer = True
        elif  color == 1 or color == 'black':
            self.maximizingPlayer = False
            
    def playTurn(self, boardPosition):
        evaluation, newBestState, bestMove = self.treeSearch.minimax(boardPosition, 
                                                self.depth, self.maximizingPlayer)
        return bestMove
        

# def main():
    # match = matchState()
    # match.startTheGame()
    

# if __name__ == "__main__":
    # main()




































