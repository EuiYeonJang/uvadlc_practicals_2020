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
        self.W_gx = nn.Parameter(torch.empty(input_dim, hidden_dim), requires_grad=True) 
        self.W_ix = nn.Parameter(torch.empty(input_dim, hidden_dim), requires_grad=True) 
        self.W_fx = nn.Parameter(torch.empty(input_dim, hidden_dim), requires_grad=True) 
        self.W_ox = nn.Parameter(torch.empty(input_dim, hidden_dim), requires_grad=True) 

        self.W_gh = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True) 
        self.W_ih = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True) 
        self.W_fh = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True) 
        self.W_oh = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True) 

        # hidden * hidden <= 4
        # input * hidden <= 4
        # output * hidden 
        # hidden <= 4
        # output
        # total = hidden( 4 (input + hidden + 1) +  output) + output

        # biases
        self.b_g = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim, 1), requires_grad=True)

        self.W_ph = nn.Parameter(torch.empty(hidden_dim, num_classes), requires_grad=True)
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1), requires_grad=True)

        self.init_params()

        # non-linearity
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def init_params(self):
        nn.init.kaiming_normal_(self.W_gx)
        nn.init.kaiming_normal_(self.W_ix)
        nn.init.kaiming_normal_(self.W_fx)
        nn.init.kaiming_normal_(self.W_ox)
        nn.init.kaiming_normal_(self.W_gh)
        nn.init.kaiming_normal_(self.W_ih)
        nn.init.kaiming_normal_(self.W_fh)
        nn.init.kaiming_normal_(self.W_oh)
        nn.init.kaiming_normal_(self.W_ph)


    def forward(self, x, c, h):
        # Eq 4d
        g = self.tanh(self.W_gx(x) + self.W_gh(h) + self.b_g)
        # Eq 5
        i = self.sigmoid(self.ix(x) + self.W_ih(h) + self.b_i)
        # Eq 6
        f = self.sigmoid(self.W_fx(x) + self.W_fh(h) + self.b_f)
        # Eq 7
        o = self.sigmoid(self.W_ox(x) + self.W_oh(h) + self.b_o)

        # Eq 8
        c = g * i + c * f
        # Eq 9
        h = self.tanh(c) * o

        # Eq 10
        p = self.W_oh(h)
        p += self.b_p
        y = self.softmax(p)
        return y, c, h


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length

        self.cell = LSTMCell(input_dim, hidden_dim, num_classes, device)
        
        self.c = torch.empty(hidden_dim, hidden_dim)
        self.h = torch.empty(hidden_dim, hidden_dim)

        self.init_params()
        ########################
        # END OF YOUR CODE    #
        #######################

    def init_params(self):
        nn.init.kaiming_normal_(self.c) #, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.h) #, mode='fan_in', nonlinearity='relu')
        

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        for t in range(self.seq_length):
            y, self.c, self.h = self.cell(x[:,:,t], self.c, self.h)

        return y
        ########################
        # END OF YOUR CODE    #
        #######################
