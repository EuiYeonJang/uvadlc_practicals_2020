"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self._seq_length = seq_length
        self._hidden_dim = hidden_dim
        self._batch_size = batch_size
        self._device = device

        embedding_size = 10 # as a starting point
        self.embed = nn.Embedding(input_dim, embedding_size)

        # weights
        self.W_gx = nn.Parameter(torch.empty(hidden_dim, embedding_size)) 
        self.W_ix = nn.Parameter(torch.empty(hidden_dim, embedding_size)) 
        self.W_fx = nn.Parameter(torch.empty(hidden_dim, embedding_size)) 
        self.W_ox = nn.Parameter(torch.empty(hidden_dim, embedding_size)) 

        self.W_gh = nn.Parameter(torch.empty(hidden_dim, hidden_dim)) 
        self.W_ih = nn.Parameter(torch.empty(hidden_dim, hidden_dim)) 
        self.W_fh = nn.Parameter(torch.empty(hidden_dim, hidden_dim)) 
        self.W_oh = nn.Parameter(torch.empty(hidden_dim, hidden_dim)) 

        self.W_ph = nn.Parameter(torch.empty(num_classes, hidden_dim))
        
        # biases
        self.b_g = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim, 1))

        self.b_p = nn.Parameter(torch.zeros(1, num_classes))

        # non-linearity
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        ########################
        # END OF YOUR CODE    #
        #######################

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.startswith("W_"):
                nn.init.kaiming_normal_(param, nonlinearity="linear")

    def init_states(self): 
        self.c = torch.zeros(self._batch_size, self._hidden_dim).to(self._device)
        self.h = torch.zeros(self._batch_size, self._hidden_dim).to(self._device)

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.squeeze() # we don't need the featuer dim, it will always be 1
        self.init_states() # initialise states each time

        embed_x = self.embed(x.long())

        for t in range(self._seq_length):
            # Eq 4
            x_t = embed_x[:, t]
            g = self.tanh(x_t@self.W_gx.T + self.h@self.W_gh.T + self.b_g)
            # Eq 5
            i = self.sigmoid(x_t@self.W_ix.T + self.h@self.W_ih.T + self.b_i)
            # Eq 6
            f = self.sigmoid(x_t@self.W_fx.T + self.h@self.W_fh.T + self.b_f)
            # Eq 7
            o = self.sigmoid(x_t@self.W_ox.T + self.h@self.W_oh.T + self.b_o)

            # Eq 8
            self.c = g * i + self.c * f
            # Eq 9
            self.h = self.tanh(self.c) * o

            # Eq 10
            y = self.h@self.W_ph.T + self.b_p

        return y # no need for log softmax since using cross entropy
        ########################
        # END OF YOUR CODE    #
        #######################
