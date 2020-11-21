"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, device):
        super().__init__()

        # Weights
        self.W_gx = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 
        self.W_ix = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 
        self.W_fx = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 
        self.W_ox = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 

        self.W_gh = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 
        self.W_ih = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 
        self.W_fh = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 
        self.W_oh = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True).to(device) 

        # biases
        self.b_g = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True).to(device)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True).to(device)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True).to(device)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True).to(device)

        self.W_ph = nn.Parameter(torch.empty(num_classes, hidden_dim), requires_grad=True).to(device)
        self.b_p = nn.Parameter(torch.zeros(1, num_classes), requires_grad=True).to(device)

        self.init_params()

        # non-linearity
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)


    def init_params(self):
        nn.init.kaiming_normal_(self.W_gx, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_ix, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_fx, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_ox, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_gh, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_ih, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_fh, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_oh, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W_ph, nonlinearity="linear")


    def forward(self, x, c, h):
        # Eq 4d
        g = self.tanh(x.T @ self.W_gx + h@self.W_gh.T + self.b_g)
        # Eq 5
        i = self.sigmoid(x.T @ self.W_ix + h@self.W_ih.T + self.b_i)
        # Eq 6
        f = self.sigmoid(x.T @ self.W_fx + h@self.W_fh.T + self.b_f)
        # Eq 7
        o = self.sigmoid(x.T @ self.W_ox + h@self.W_oh.T + self.b_o)

        # Eq 8
        c = g * i + c * f
        # Eq 9
        h = self.tanh(c) * o

        # Eq 10
        p = h@self.W_ph.T + self.b_p

        # y = self.softmax(p)

        return p, c, h


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.device = device
        embedding_size = int(hidden_dim/2) 
        # self.embedding = nn.Embedding(input_dim, embedding_size, padding_idx=1)
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=1)

        self.cell = LSTMCell(input_dim, hidden_dim, num_classes, device)
        
        self.c = torch.empty(hidden_dim, hidden_dim).to(device)
        self.h = torch.empty(hidden_dim, hidden_dim).to(device)

        self.init_params()
        ########################
        # END OF YOUR CODE    #
        #######################

    def init_params(self):
        nn.init.kaiming_normal_(self.c, nonlinearity="linear") #, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.h, nonlinearity="linear") #, mode='fan_in', nonlinearity='relu')
        

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.squeeze()
        self.c = torch.empty(self.hidden_dim, self.hidden_dim).to(self.device)
        self.h = torch.empty(self.hidden_dim, self.hidden_dim).to(self.device)
        self.init_params()

        embed_x = self.embedding(x.long())
        for t in range(self.seq_length):
            # y, self.c, self.h = self.cell(embed_x[:,:,t], self.c, self.h)
            y, self.c, self.h = self.cell(embed_x[:,t], self.c, self.h)


        return y
        ########################
        # END OF YOUR CODE    #
        #######################
