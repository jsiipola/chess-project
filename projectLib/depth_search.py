#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:13:44 2023

@author: siipola
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import sys


class DepthSearcher(object):
    """ Aim of this class is to provide tool that is able to search positions
        in the depth of the game. """
    def __init__(self, evaluationFunction, featureRep, device):
        """ Position wherefrom the search begins. 
            And for which the decision of a move is made. """
        # self.board_root_position = board_root_position
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
        """ First check if the depth is zero or if the game has ended. If so,
            return the evaluation at the current position. Also return
            the next state"""
        checkmate = self.featureRepresentation.detect_check_mate(position)
        if depth == 0 or checkmate is True:
            position_features = self.featureRepresentation.feature_representation(position)
            return self.evaluationFunction.evaluate(position_features), None, None
        """ If the player choosing the move is maximizing player, then go here. """
        if maximizingPlayer:
            """ maxEval is going to be the output of this level of nodes, and 
                it is initialized as minus infinity. """
            maxEval = -torch.tensor(float('inf'))
            """ As the depth was not zero, then this node has child nodes, we
                list the child node positions and do the evaluation of the position
                for each of those. """
            listOfMoveChildren, moves = self.featureRepresentation.getChildPositions(position)
            t = -1
            for positionChild in listOfMoveChildren:
                t += 1
                evaluation, comingUpNextState, tmp_move = self.minimax(positionChild, depth-1, False)
                if maxEval < evaluation:
                    newBestState = positionChild
                    bestMove = moves[t]
                maxEval = torch.maximum(maxEval, evaluation)
            # print('maximizingPlayer')
            # print(newBestState)
            # print(maxEval)
            return maxEval, newBestState, bestMove
        else:
            minEval = torch.tensor(float('inf'))
            listOfMoveChildren, moves = self.featureRepresentation.getChildPositions(position)
            t = -1
            for positionChild in listOfMoveChildren:
                t += 1
                evaluation, comingUpNextState, tmp_move = self.minimax(positionChild, depth-1, True)
                if minEval > evaluation:
                    newBestState = positionChild
                    bestMove = moves[t]
                minEval = torch.minimum(minEval, evaluation)
            # print('minimizingPlayer')
            # print(newBestState)
            # print(minEval)
            return minEval, newBestState, bestMove
        
    def minimax_alphaBetaPruning(self, position, depth, maximizingPlayer, 
                                 alpha = -torch.tensor(float("inf")), 
                                 beta  = +torch.tensor(float("inf"))):
        """ args:
                position = board position, not the feature representation. 
                depth = an integer, tells how deep into the tree the search is 
                            supposed to dive into.
                maximizingPlayer = True or False, if True then the algorithm 
                        think we are starting from the position where the 
                        maximizing player is choosing the next move. 
                alpha, beta = internal parameters of the method, value should
                        be a real number, in practice a torch tensor with single
                        value. In the initial function call alpha should be given
                        value -infinity, and beta +infinity.
            
            pseudocode from Sebastian Lague's youtube channel: 
                            https://www.youtube.com/watch?v=l-hh51ncgDI
                        """
        """ First check if the depth is zero or if the game has ended. If so,
            return the evaluation at the current position. """
        checkmate = self.featureRepresentation.detect_check_mate(position)
        if depth == 0 or checkmate is True:
            position_features = self.featureRepresentation.feature_representation(position)
            output = self.evaluationFunction.evaluate(position_features)
            return output
        
        """ If the player choosing the move is maximizing player, then go here. """
        if maximizingPlayer:
            """ maxEval is going to be the output of this level of nodes, and 
                it is initialized as minus infinity. """
            maxEval = -torch.tensor(float('inf'))
            """ As the depth was not zero, then this node has child nodes, we
                list the child node positions and do the evaluation of the position
                for each of those. """
            listOfMoveChildren = self.featureRepresentation.getChildPositions(position)
            for positionChild in listOfMoveChildren:
                evaluation = self.minimax_alphaBetaPruning(positionChild, depth-1, 
                                                           False, alpha, beta)
                maxEval = torch.maximum(maxEval, evaluation)
                """ For alpha beta pruning we have this extra moment here with
                    alpha and beta parameters. """
                alpha = torch.maximum(alpha, evaluation)
                if beta <= alpha:
                    break
            return maxEval
        # """ In the case the player is minimizing player then go here. """
        else:
            """ Similiar process as in the case of maximizing player, except that
            here we start from positive infinity and look for minimal evaluation. """
            minEval = torch.tensor(float('inf'))
            listOfMoveChildren = self.featureRepresentation.getChildPositions(position)
            for positionChild in listOfMoveChildren:
                evaluation = self.minimax_alphaBetaPruning(positionChild, depth-1, 
                                                           True, alpha, beta)

                minEval = torch.minimum(minEval, evaluation)
                beta = torch.minimum(beta, evaluation)
                previousDebugChild = positionChild
                if beta <= alpha:
                    break
            return minEval
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

























































































































