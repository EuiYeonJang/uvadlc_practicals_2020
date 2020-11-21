"""
This module implements a GRU in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(GRU, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = device

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        embedding_size = 10
        self.embed = nn.Embedding(input_dim, embedding_size)

        weight_size = embedding_size + hidden_dim
        self.W_z = nn.Parameter(torch.empty(hidden_dim, weight_size))
        self.W_r = nn.Parameter(torch.empty(hidden_dim, weight_size))
        self.W = nn.Parameter(torch.empty(hidden_dim, weight_size))
        self.W_ph = nn.Parameter(torch.empty(num_classes, hidden_dim))

        self.b_p = nn.Parameter(torch.zeros(num_classes))

        self.init_weights()
        ########################
        # END OF YOUR CODE    #
        #######################

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.startswith("W"):
                nn.init.kaiming_normal_(param, nonlinearity="linear")


    def init_states(self):
        self.c = torch.zeros(self._batch_size, self._hidden_dim).to(self._device)
        self.h = torch.zeros(self._batch_size, self._hidden_dim).to(self._device)


    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.init_states()
        x = x.squeeze()
        embed_x = self.embed(x.long())
        for t in range(self._seq_length):
            x_t = embed_x[:, t]
            
            # Eq 29
            h_prev_x_t = torch.cat((self.h, x_t), 1)
            z = self.sigmoid(h_prev_x_t@self.W_z.T)
            # Eq 30
            r = self.sigmoid(h_prev_x_t@self.W_r.T)
            
            # Eq 31
            r_h_prev_x_t = torch.cat((self.h*r, x_t), 1)
            h_tilde = self.tanh(r_h_prev_x_t@self.W.T)
            
            # Eq 32
            self.h = (1 - z)*self.h + z*h_tilde

            # Eq 33
            p = self.h@self.W_ph.T + self.b_p

        # no need for softmax since using cross entropy
        return p

        ########################
        # END OF YOUR CODE    #
        #######################
