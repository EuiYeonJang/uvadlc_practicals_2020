"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

N_FEATURES = 32 * 32* 3
N_CLASSES = 10

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
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = (pred_classes==targets).sum().item() / len(targets)
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy

def param_init(model, std=0.0001):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            param.data.normal_(std=std)


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    dataset = cifar10_utils.get_cifar10(one_hot=False)

    model  = MLP(N_FEATURES, dnn_hidden_units, N_CLASSES)
    param_init(model)
    print(model)

    loss_module = nn.CrossEntropyLoss()
    sm = nn.Softmax(dim=1)
    # optimiser = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9)
    optimiser = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    # momentum 0.9 and layers 1000,100 stays around 0.53 but goes up to 0.54
    # momentum 0.9 and layers 500,100 with ReLU also goes around 0.52
    # 
    model.train()

    for step in range(FLAGS.max_steps):
        x, y = dataset["train"].next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x.reshape(FLAGS.batch_size, -1)).to(device)
        y = torch.from_numpy(y).to(device)

        preds = model(x)

        loss = loss_module(preds, y.long())

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

        if step % FLAGS.eval_freq == 0:
            model.eval()

            with torch.no_grad():
                x_test, y_test = dataset["test"].images, dataset["test"].labels
                x_test = torch.from_numpy(x_test.reshape(x_test.shape[0], -1)).to(device)
                y_test = torch.from_numpy(y_test).to(device)

                preds_test = model(x_test)
                preds_test = sm(preds_test)

                acc = accuracy(preds_test, y_test)
                print(f"step {step} -- ACC {acc:.2f}")
                # return
            
            model.train()
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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
