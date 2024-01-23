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
    def __init__(self, evaluationFunction, featureRep, device, neural_net):
        """ Position wherefrom the search begins. 
            And for which the decision of a move is made. """
        # self.board_root_position = board_root_position
        self.evaluationFunction = evaluationFunction
        self.featureRepresentation = featureRep
        self.device = device
        self.neural_net = neural_net

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
            return the evaluation at the current position. """
        checkmate = self.featureRepresentation.detect_check_mate(position)
        if depth == 0 or checkmate is True:
            position_features = self.featureRepresentation.feature_representation(position)
            return self.evaluationFunction.evaluate(position_features)
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
                evaluation = self.minimax(positionChild, depth-1, False)
                maxEval = torch.maximum(maxEval, evaluation)
            return maxEval
        else:
            minEval = torch.tensor(float('inf'))
            listOfMoveChildren = self.featureRepresentation.getChildPositions(position)
            for positionChild in listOfMoveChildren:
                evaluation = self.minimax(positionChild, depth-1, True)
                minEval = torch.minimum(minEval, evaluation)
            return minEval
        
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
            # print('evaluation')
            # print(evaluation)
            # print(evaluation.item())
            # if torch.isnan(output).item():
                # print('position')
                # print(position)
                # print('previousDebugChild')
                # print(previousDebugChild)
                # position_feature = self.featureRepresentation.feature_representation(positionChild)
                # print('position_feature')
                # print(position_features)
                # position_featureDebugChild = self.featureRepresentation.feature_representation(previousDebugChild)
                # print('position_featureDebugChild')
                # print(position_featureDebugChild)
                # self.neural_net.forwardPublic(position_features)
                # self.neural_net.forwardPublic(position_featureDebugChild)
                # sys.exit()
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
                # print('evaluation')
                # print(evaluation)
                # print(evaluation.item())
                # print(torch.isnan(evaluation).item())
                # previousDebugChild = positionChild
                # if torch.isnan(evaluation).item():
                    # print('positionChild')
                #     print(positionChild)
                #     print('previousDebugChild')
                #     print(previousDebugChild)
                #     position_feature = self.featureRepresentation.feature_representation(positionChild)
                #     print('position_feature')
                #     print(position_feature)
                #     position_featureDebugChild = self.featureRepresentation.feature_representation(previousDebugChild)
                #     print('position_featureDebugChild')
                #     print(position_featureDebugChild)
                #     self.neural_net.forwardPublic(position_feature)
                #     self.neural_net.forwardPublic(position_featureDebugChild)
                #     sys.exit()
                # print('depth')
                # print(depth)
                minEval = torch.minimum(minEval, evaluation)
                beta = torch.minimum(beta, evaluation)
                previousDebugChild = positionChild
                if beta <= alpha:
                    break
            return minEval
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

























































































































