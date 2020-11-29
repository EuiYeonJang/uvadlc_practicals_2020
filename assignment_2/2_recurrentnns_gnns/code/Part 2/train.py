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
import pickle as pkl

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################

def scale(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # FIXME
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, device).to(device)

    print("Loading model")
    state_dict = torch.load(f"{config.summary_path}trained_dem.mdl")
    model.load_state_dict(state_dict)
    
    print("Temperature sampling.")
    
    n_samples = 3
    temperature_sent = list()
    model.eval()

    softmax = torch.nn.Softmax(dim=-1)

    for n in range(n_samples):
        output_idx = init_idx = np.random.randint(dataset.vocab_size)
        
        samples = dict()
        
        for tao in [0.5, 1, 2]:
            gen_txt = [init_idx]
            for t in range(config.seq_length):
                if t == 0:
                    idx = torch.LongTensor([[output_idx]]).to(device)
                    output, (h, c) = model(idx)
                else:
                    idx = torch.LongTensor([[output_idx]]).to(device)
                    output, (h, c) = model(idx, (h, c))
                
                distr= softmax(tao*output).squeeze()
                output_idx = torch.multinomial(distr, 1).item()
                gen_txt.append(output_idx)

            samples[tao] = dataset.convert_to_string(gen_txt)
        
        temperature_sent.append(samples)

    

    # train_data = dict(acc=acc_list, loss=loss_list, greedy_sent=greedy_sent, temperature_sent=temperature_sent)

    # with open(f"{config.summary_path}data.pkl", "wb") as f:
        # pkl.dump(train_data, f)


def bonus(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # FIXME
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, device).to(device)

    print("Loading model...")
    state_dict = torch.load(f"{config.summary_path}{config.model_name}.mdl")
    model.load_state_dict(state_dict)
    
    softmax = torch.nn.Softmax(dim=-1)

    print("Generating sentence...")
    start_sent = "Sleeping beauty is "
    finish_sent_l = []

    for i, s in enumerate(start_sent):
        s_idx = dataset._char_to_ix[s]
        s_idx = torch.LongTensor([[s_idx]]).to(device)
        if i == 0:
            _, (h, c) = model(s_idx)
        else:
            output_idx, (h, c) = model(s_idx, (h, c))

    period = dataset._char_to_ix(["."])

    while output_idx != period:
        finish_sent_l.append(output_idx)

        output, (h, c) = model(s_idx, (h, c))
        distr= softmax(2*output).squeeze()
        output_idx = torch.multinomial(distr, 1).item()

    finish_sent = dataset.convert_to_string(finish_sent_l)
    print("".join([start_sent, finish_sent]))


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
    loss_list = list()
    greedy_sent = list()
    
    model.train()

    config.sample_every = int((2*len(data_loader))/3)
    config.train_steps = len(data_loader)

    train_stage = 0

    for epoch in range(2):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()
            #######################################################
            # Add more code here ...
            #######################################################
            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).T.to(device)

            model.zero_grad()

            preds, _ = model(batch_inputs)
            preds = preds.transpose(0, 1)
            preds = preds.transpose(1, 2)
            
            loss = criterion(preds, batch_targets) # FIXME
            loss_list.append(loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)

            optimizer.step()
            
            preds = torch.argmax(preds, dim=1)
            correct = (preds == batch_targets).sum().item()
            accuracy = correct / (preds.shape[0]*preds.shape[1]) # FIXME
            acc_list.append(accuracy)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if (step + 1) % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Epoch {} Batch Size = {}, \
                        Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, epoch, config.batch_size, examples_per_second,
                        accuracy, loss
                        ))


            if (train_stage + 1) % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                n_samples = 5
                less_than = int(config.seq_length/2)
                more_than = config.seq_length*2
                model.eval()

                with torch.no_grad():

                    for n in range(n_samples):
                        rand_idx = np.random.randint(dataset.vocab_size)
                        gen_txt = [rand_idx] # select a random index to generate sentence

                        samples = dict()
                    
                        for t in range(more_than):
                            if t == 0:
                                idx = torch.LongTensor([[rand_idx]]).to(device)
                                output, (h, c) = model(idx)

                            elif t + 1 == less_than:
                                samples[less_than] = dataset.convert_to_string(gen_txt)

                            elif t +1 == config.seq_length:
                                samples[config.seq_length] = dataset.convert_to_string(gen_txt)

                            else:
                                idx = torch.LongTensor([[idx]]).to(device)
                                output, (h, c) = model(idx, (h, c))
                            
                            idx = torch.argmax(output).item()
                            gen_txt.append(idx)

                        samples[more_than] = dataset.convert_to_string(gen_txt)
                        greedy_sent.append(samples)

                    model.train()

            train_stage += 1

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
    
    print('Done training.')

    print("Saving model.")
    state_dict = model.state_dict()
    torch.save(state_dict, f"{config.summary_path}trained_grimms.mdl")

    
###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, #required=True,
                        default="./assets/book_EN_grimms_fairy_tails.txt",
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


    parser.add_argument('--load_model', type=bool, default=False,
                            help="Indicate to load saved model.")
    parser.add_argument('--model_name', type=str, default="trained_dem",
                            help="Specify to name of saved model.")
    # If needed/wanted, feel free to add more arguments
    config = parser.parse_args()

    print(config)

    # Train the model
    if not config.load_model:
        train(config)
    
    bonus(config)