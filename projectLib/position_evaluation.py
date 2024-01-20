#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:44:32 2023

@author: siipola
"""



class evaluator(object):
    def __init__(self, neural_network):
        self.net = neural_network


    def evaluate(self, board_features):
        print('hip')
        return self.net.forward(board_features)
        


























