#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:13:44 2023

@author: siipola
"""
import torch
import sys


class depth_searcher(object):
    """ Aim of this class is to provide tool that is able to search positions
        in the depth of the game. """
    def __init__(self, board_root_position, evaluationFunction, featureRep, device):
        """ Position wherefrom the search begins. 
            And for which the decision of a move is made. """
        self.board_root_position = board_root_position
        self.evaluationFunction = evaluationFunction
        self.featureRepresentation = featureRep
        self.device = device

    def update_root_position(self, board):
        """ With this method we update the root position into a new position. """
        self.board_root_position = board
    
    def minimax(self, position, depth, maximizingPlayer):
        """ args:
                position = board position, not the feature representation. 
                depth = an integer, tells how deep into the tree the search is 
                            supposed to dive into.
                maximizingPlayer = True or False, if True then the algorithm 
                        think we are starting from the position where the 
                        maximizing player is choosing the next move. 
                        
            
            pseudocode from Sebastian Lague's youtube channel: 
                            https://www.youtube.com/watch?v=l-hh51ncgDI
                        """
        checkmate = self.featureRep.detect_check_mate(position)
        if depth == 0 or checkmate is True:
            return self.evaluationFunction(position)
        if maximizingPlayer:
            maxEval = -torch.tensor(float('inf'))
            listOfMoveChildren = self.featureRepresentation.getChildPositions()
            for position in listOfMoveChildren:
                evaluation = self.minimax(position, depth-1, False)
                maxEval = torch.maximum(maxEval, evaluation)
            return maxEval
        else:
            minEval = torch.tensor(float('inf'))
            listOfMoveChildren = self.featureRepresentation.getChildPositions()
            for position in listOfMoveChildren:
                evaluation = self.minimax(position, depth-1, True)
                minEval = torch.minimum(maxEval, evaluation)
            return minEval
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

























































































































