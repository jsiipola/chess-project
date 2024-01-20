#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 23:04:17 2023

@author: siipola
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

#from . import activations
#from . import initializers
#from . import regularizers
from .map import Map
from .. import config
#from ..backend import tf
from ..utils import timing

# Pytorch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys

class FNN(Map):
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
        layer_size,
        activation = 'weinan',
#        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_normalization=None,
        biases = True,
        drop_out = None,
        drop_out_rate = 0.5
    ):
        super(FNN, self).__init__()
        self.layer_size = layer_size
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

    @timing
    def build(self):
        """
        Very core method for this FNN class. This method builds the feed
        forward network. It does not require any arguments. All the data comes
        from the attributes of the class.
        """
        print("Building feed-forward neural network...")

        
        # Using nn.ModuleList
        class neuralNet(nn.Module):
            def __init__(self, layer_size, biases, drop_out, drop_out_rate,
                         activation):
                """
                args:
                    layer_size: For example [3,20,20,20,3].
                    biases: True of False. Tells if we want to use bias terms in
                            the layers of the neural net.
                """
                super(neuralNet, self).__init__()
                self.layers = nn.ModuleList()
                for i in range(len(layer_size) - 1):
                    self.layers.append(nn.Linear(layer_size[i], layer_size[i+1], biases))
                
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
                    self.activation = nn.ReLu()
                elif activation == 'tanh':
                    self.activation = nn.Tanh()
                elif activation == 'Softplus':
                    self.activation = nn.Softplus()
                    
            def forward(self, x):
                """
                At the moment this forward method assumes only relu as activation
                function.
                args:
                    x: The input vector of the neural net.
                """

                
                """ Same activation on all layers. """
                # m = nn.ReLU()
                # m = nn.Tanh()
                # m = nn.Softplus()
                
                for i in range(len(self.layers)-1):
                    """ Same activation on all layers: """
                    
                    x = self.activation(self.layers[i](x))
                    # x = self.dropout_layer(x)
   
                x = self.layers[len(self.layers)-1](x)
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
        
        """ Next line builds the neural net. """
        self.neural_net = neuralNet(self.layer_size, self.biases, self.drop_out, 
                                    self.drop_out_rate, self.activation)
        self.built = True

        




""" archive """
                
""" If one want to use the partly softplus partly tanh activation
    then uncomment the next if-else-block and use it. """
# if (x.dim() == 2):
#     for i in range(len(self.layers)):
#         """ Same activation on all layers: """
#         # m = nn.Sigmoid()
#         m2 = nn.Tanh()
#         m = nn.Softplus()
#         # m = nn.ReLU()
#         # if (i%2==0):
#             # m = nn.Softplus()
#         # else:
#             # m = nn.Tanh()
#         if (i==0 or i == (len(self.layers)-1)):
#             x = m(self.layers[i](x))
#         else:
#             # print(i)
#             split_point = 20
#             linear_out = self.layers[i](x) # shape is (batches, features)
#             # print('linear_out.size()')
#             # print(linear_out.size())
#             first_slice = m(linear_out[:,0:split_point])
#             second_slice = m2(linear_out[:,split_point:])
#             x = torch.cat((first_slice, second_slice), dim=1)
# else:
#     for i in range(len(self.layers)):
#         """ Same activation on all layers: """
#         # m = nn.Sigmoid()
#         m2 = nn.Tanh()
#         m = nn.Softplus()
#         # m = nn.ReLU()
#         # if (i%2==0):
#             # m = nn.Softplus()
#         # else:
#             # m = nn.Tanh()
#         if (i==0 or i == (len(self.n mukaansa puoliso ei tiennyt edes kotona olevan aseita, eikä olisi hyväksynyt asiaa. layers)-1)):
#             x = m(self.layers[i](x))
#         else:
#             # print(i)
#             split_point = 20
#             linear_out = self.layers[i](x) # shape is (batches, features)
#             # print('linear_out.size()')
#             # print(linear_out.size())
#             first_slice = m(linear_out[0:split_point])
#             second_slice = m2(linear_out[split_point:])
#             x = torch.cat((first_slice, second_slice))

""" Residual net. """
# m = nn.ReLU()
# # m = nn.Softplus()
# for i in range(len(self.layers)):
#     if (i == 0):
#         x = self.layers[i](x)
#     elif (i%2 == 1): #(residuaalin lähtö):
#         x_res = x.detach().clone()
#         x = m(self.layers[i](x))
#         # print('@ fnn line 162')
#         # print(x.shape)
#     else: #(residuaalinen paluu)
#         x = m(self.layers[i](x)) + x_res
    # """ Same activation on all layers: """
    # x = m(self.layers[i](x))
    # a = self.layers[i](x)
    # print('@ fnn line 162')
    # print(a)
    # x = self.weinan_activation(a)
    # print('@ fnn line 165')
    # print(x)
    # sys.exit()
    
        
# class readyMadeNeuralNet(nn.Module):
#     def __init__(self):
#         super(readyMadeNeuralNet, self).__init__()
#         """
#         Initialization of the neural net. 
#         """
#         self.fc1 = nn.Linear(1, 10)
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 1)
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return x
