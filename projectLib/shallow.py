#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:58:23 2023

@author: siipola
"""


from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import math

#from . import activations
#from . import initializers
#from . import regularizers
#from .map import Map
#from .. import config
#from ..backend import tf
#from ..utils import timing

# Pytorch
import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np
#import torch
import sys

class shallow(object):
    """
    Class for Feed-forward Neural Networks, put it shortly FNN.
    
    args:
        layer_size: How many layers and how many nodes in each layer.
        activation: what activation function to be used for the layers.
        kernel_initializer:
        regularization:
        dropout_rate:
        batch_normalization:
    """

    def __init__(
        self,
        layer_width,
        activation,
        N_features,
#        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_normalization=None,
        biases = True,
        drop_out = None,
        drop_out_rate = 0.5
        
    ):
        super(shallow, self).__init__()
        self.layer_width = layer_width
        self.activation = activation
#        self.kernel_initializer = initializers.get(kernel_initializer)
#        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.neural_net = None
        self.biases = biases
        self.built = False
        self.drop_out = drop_out
        self.drop_out_rate = drop_out_rate
        self.N_features = N_features
        
    @property
    def inputs(self):
        return self.x
#
#    @property
#    def outputs(self):
#        return self.y
#
#    @property
#    def targets(self):
#        return self.y_

#    @timing
    def build(self):
        print("Building feed-forward neural network...")

        
        # Using nn.ModuleList
        class neuralNet(nn.Module):
            def __init__(self, layer_width, biases, drop_out, drop_out_rate,
                         activation, N_features):
                """
                args:
                    layer_size: For example [3,20,20,20,3].
                    biases: True of False. Tells if we want to use bias terms in
                            the layers of the neural net.
                """
                super(neuralNet, self).__init__()
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(N_features, layer_width, biases))
                self.layers.append(nn.Linear(layer_width, 1, biases))
                
                """ Add possible dropout to the model. """
                if drop_out == True:
                    self.dropout_layer = nn.Dropout(p = drop_out_rate)
                else:
                    self.dropout_layer = nn.Dropout(p = 0)
                    
                """ Define the activation function """
                if activation == 'weinan' or activation == None:
                    self.activation = self.weinan_activation
                elif activation == 'swiss':
                    self.activation = self.swish_activation()
                elif activation == 'relu':
                    self.activation = nn.ReLU()
                elif activation == 'tanh':
                    self.activation = nn.Tanh()
                elif activation == 'Softplus':
                    self.activation = nn.Softplus()
                elif activation == 'gelu':
                    self.activation = nn.GELU()
                    
            def forward(self, x):
                """
                At the moment this forward method assumes only relu as activation
                function.
                args:
                    x: The input vector of the neural net.
                """
                x = self.activation(self.layers[0](x))  
                x = self.layers[1](x)
                return x
            
            def forwardPublic(self, x):
                """
                At the moment this forward method assumes only relu as activation
                function.
                args:
                    x: The input vector of the neural net.
                    
                For debugging. Print out some data while pushing forward in the 
                network.
                """
                x = self.activation(self.layers[0](x))  
                print('hidden: ')
                print(x)
                x = self.layers[1](x)
                print('output')
                print(x)
                return x
            
            
            
            def weinan_activation(self, x):
                """ Activation function defined in paper:
                    The Deep Ritz method: A deep learning_based numerical algorithm
                    for solving variational problems,
                    Weinan E., Bing Yu """
                m = nn.ReLU()
                x = m(x)**2
                return x
        
            def weinan_like_activation(self, x):
                m = nn.ReLU()
                x = (1.0/6.0)*m(x)**3
                return x                
            def swish_activation(self, x):
                """ Swish: a self-gated activation function. """
                m = nn.Sigmoid()
                x = x*m(x)
                return x
            print('Build succesfull.')
        
        """ Next line builds the neural net. """
        self.neural_net = neuralNet(self.layer_width, self.biases, self.drop_out, 
                                    self.drop_out_rate, self.activation, 
                                    self.N_features)
        self.built = True
