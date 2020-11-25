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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # FIXME
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, device).to(device)  # FIXME

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # FIXME

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # FIXME

    acc_list = list()
    
    model.train()

    for epoch in range(1):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()
            #######################################################
            # Add more code here ...
            #######################################################
            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).to(device)

            model.zero_grad()

            preds, _ = model(batch_inputs)
            preds = preds.transpose(0, 1)
            preds = preds.transpose(1, 2)   
            batch_targets = batch_targets.T
            
            loss = criterion(preds, batch_targets) # FIXME
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)

            optimizer.step()
            
            preds = torch.argmax(preds, dim=1)
            correct = (preds == batch_targets).sum().item()
            accuracy = correct / (config.batch_size*config.seq_length) # FIXME
        
            acc_list.append(accuracy)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            
            if (step + 1) % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                        Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                        ))

            if (step + 1) % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                rand_idx = np.random.randint(dataset.vocab_size)
                gen_txt = [rand_idx]

                model.eval()

                for t in range(config.seq_length):
                    if t == 0:
                        idx = torch.tensor([[rand_idx]]).to(device)
                        output, (h, c) = model(idx)
                    else:
                        output, (h, c) = model(torch.tensor([[idx]]), (h, c))
                    
                    idx = torch.argmax(output).item()
                    gen_txt.append(idx)

                print(dataset.convert_to_string(gen_txt))

                model.train()

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, #required=True,
                        default="./assets/book_EN_democracy_in_the_US.txt",
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments
    config = parser.parse_args()
    # Train the model
    train(config)
