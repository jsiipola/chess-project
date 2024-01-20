#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 23:20:53 2023

@author: siipola
"""

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


# Using nn.ModuleList
class neuralNet(nn.Module):
    def __init__(self):
        """
        args:
            layer_size: For example [3,20,20,20,3].
            biases: True of False. Tells if we want to use bias terms in
                    the layers of the neural net.
        """
        super(neuralNet, self).__init__()
#        self.layers = nn.ModuleList()
#        for i in range(len(layer_size) - 1):
#            self.layers.append(nn.Linear(layer_size[i], layer_size[i+1], biases))
#        
#        """ Add possible dropout to the model. """
#        if drop_out == True:
#            self.dropout_layer = nn.Dropout(p = drop_out_rate)
#        else:
#            self.dropout_layer = nn.Dropout(p = 0)
#            
#        """ Define the activation function """
#        if activation == 'weinan' or activation == None:
#            self.activation = self.weinan_activation
#        elif activation == 'swiss':
#            self.activation = self.swish_activation()
#        elif activation == 'relu':
#            self.activation = nn.ReLu()
#        elif activation == 'tanh':
#            self.activation = nn.Tanh()
#        elif activation == 'Softplus':
#            self.activation = nn.Softplus()
        self.fc1 = nn.Linear(75, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
            
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

