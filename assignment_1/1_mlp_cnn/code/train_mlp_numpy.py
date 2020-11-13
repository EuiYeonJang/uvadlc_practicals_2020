"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

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
    predictions = np.argmax(predictions, axis=1)
    targets = np.argmax(targets, axis=1)
    accuracy = (predictions==targets).sum() / len(targets)
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

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
    model = MLP(N_FEATURES, dnn_hidden_units, N_CLASSES)
    train_accs = list()
    train_losses = list()
    test_accs = list()
    test_losses = list()
    loss_module = CrossEntropyModule()

    dataset = cifar10_utils.get_cifar10()

    for step in range(FLAGS.max_steps):
        x, y = dataset["train"].next_batch(FLAGS.batch_size)
        x = x.reshape(FLAGS.batch_size, -1)

        preds = model.forward(x)
        
        loss = loss_module.forward(preds, y)

        acc = accuracy(preds, y)

        loss_dx = loss_module.backward(preds, y)
        model.backward(loss_dx)

        for i in range(len(model.list_modules)):
            if i % 2 == 0:
                model.list_modules[i].params["weight"] -= FLAGS.learning_rate * model.list_modules[i].grads["weight"]
                model.list_modules[i].params["bias"] -= FLAGS.learning_rate * model.list_modules[i].grads["bias"]

        if step % FLAGS.eval_freq == 0:
            train_losses.append(loss)
            train_accs.append(acc)

            x_test, y_test = dataset["test"].images, dataset["test"].labels
            x_test = x_test.reshape(x_test.shape[0], -1)

            preds_test = model.forward(x_test)
            test_losses.append(loss_module.forward(preds_test, y_test))
            test_acc = accuracy(preds_test, y_test)
            test_accs.append(test_acc)
            print(f"step {step} -- acc {test_acc:.2f}")


    # test_x = [i for i in range(FLAGS.eval_freq, FLAGS.max_steps+1, FLAGS.eval_freq)]
    plt.plot(train_losses, c="b", label="Train Loss")
    plt.plot(test_losses, c="g", label="Test Loss")
    plt.legend(loc="upper left")
    plt.title("Train and Test Losses of Numpy MLP")
    plt.ylabel("Loss")
    plt.xlabel("Number of Steps")

    plt.savefig("./numpy_losses.pdf")

    plt.plot(train_accs, c="b", label="Train Accuracy")
    plt.plot(test_accs, c="g", label="Test Accuracy")
    plt.legend(loc="upper left")
    plt.title("Train and Test Accuracies of Numpy MLP")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Steps")

    plt.savefig("./numpy_accuracies.pdf")
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
