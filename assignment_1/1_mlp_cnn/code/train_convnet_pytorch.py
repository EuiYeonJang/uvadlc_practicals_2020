"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

N_CHANNELS = 3
N_CLASSES = 10


def plot(train_list, test_list, name):
    x_ind = [i for i in range(0, FLAGS.max_steps, FLAGS.eval_freq)]
    select_train_list = [train_list[i] for i in x_ind]
    plt.plot(train_list, c="tab:cyan", label=f"Train {name} (all)")
    plt.plot(x_ind, select_train_list, c="tab:blue", label=f"Train {name}")
    plt.plot(x_ind, test_list, c="tab:green", label=f"Test {name}")
    if name == "Loss":
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")
    plt.title(f"Train and Test {name} of ConvNet")
    plt.ylabel(name)
    plt.xlabel("Number of Steps")

    plt.savefig(f"./convnet_{name.lower()}.pdf")
    plt.clf()


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    pred_classes = torch.argmax(predictions, dim=1)
    accuracy = (pred_classes==targets).sum().item() / len(targets)
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    dataset = cifar10_utils.get_cifar10(one_hot=False)
    
    model = ConvNet(N_CHANNELS, N_CLASSES).to(device)
    print(model)

    loss_module = nn.CrossEntropyLoss()
    sm = nn.Softmax(dim=1)

    optimiser = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    model.train()

    train_losses = list()
    test_losses = list()
    train_accs = list()
    test_accs = list()

    for step in range(FLAGS.max_steps):
        x, y = dataset["train"].next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        preds = model(x)

        loss = loss_module(preds, y.long())
        acc = accuracy(preds, y)

        train_losses.append(loss)
        train_accs.append(acc)
        
        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

        if step % FLAGS.eval_freq == 0:
            model.eval()

            with torch.no_grad():
                x_test, y_test = dataset["test"].images, dataset["test"].labels
                x_test = torch.from_numpy(x_test).to(device)
                y_test = torch.from_numpy(y_test).to(device)

                preds_test = model(x_test)

                test_loss = loss_module(preds_test, y_test.long())

                preds_test = sm(preds_test)
                test_acc = accuracy(preds_test, y_test)
                
                test_losses.append(test_loss)
                test_accs.append(test_acc)

                print(f"step {step} -- train ACC {acc:.2f} -- test ACC {test_acc:.2f}")
            model.train()

    plot(train_losses, test_losses, "Loss")
    plot(train_accs, test_accs, "Accuracy")
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
