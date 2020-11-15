"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class PreActResNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.preact_net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        preact = self.preact_net(x)

        return x + preact


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        channels = [n_channels, 64, 128, 256, 512]

        MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ReLU = nn.ReLU()

        conv0 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)

        preact1 = PreActResNet(channels[1])

        conv1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=1),
            MaxPool
        )

        preact2a = PreActResNet(channels[2])
        
        preact2b = PreActResNet(channels[2])

        conv2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=1),
            MaxPool
        )

        preact3a = PreActResNet(channels[3])
        
        preact3b = PreActResNet(channels[3])

        conv3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=1),
            MaxPool
        )

        preact4a = PreActResNet(channels[4])
        
        preact4b = PreActResNet(channels[4])

        preact5a = PreActResNet(channels[4])
        
        preact5b = PreActResNet(channels[4])

        last = nn.Sequential(
            MaxPool,
            nn.BatchNorm2d(channels[4]),
            ReLU,
            nn.Flatten(),
            nn.Linear(channels[4], n_classes)
        )

        self.list_modules = [
            conv0,
            preact1, conv1,
            preact2a, preact2b, conv2,
            preact3a, preact3b, conv3,
            preact4a, preact4b, MaxPool,
            preact5a, preact5b, last
        ]

        self.list_modules = nn.ModuleList(self.list_modules)

        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for m in self.list_modules:
            x = m(x)

        out = x
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
