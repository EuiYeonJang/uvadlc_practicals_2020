# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        embedding_size = 10
        self.embed = nn.Embedding(vocabulary_size, embedding_size)
        
        self.lstm = nn.LSTM(embedding_size, lstm_num_hidden, lstm_num_layers)

        self.output_layer = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x, prev_state=None):
        embed_x = self.embed(x)
        
        if prev_state == None:
            p, prev_state = self.lstm(embed_x) # when not provided, both h_0 and c_0 default to zero
        else:
            p, prev_state = self.lstm(embed_x, prev_state)

        logits = self.output_layer(p) # no softmax since using CrossEntropyLoss

        return logits, prev_state
