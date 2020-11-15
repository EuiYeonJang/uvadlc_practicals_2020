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

        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ReLU = nn.ReLU()

        self.conv0 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)

        self.preact1 = PreActResNet(channels[1])

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=1),
            self.MaxPool
        )

        self.preact2a = PreActResNet(channels[2])
        
        self.preact2b = PreActResNet(channels[2])

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=1),
            self.MaxPool
        )

        self.preact3a = PreActResNet(channels[3])
        
        self.preact3b = PreActResNet(channels[3])

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=1),
            self.MaxPool
        )

        self.preact4a = PreActResNet(channels[4])
        
        self.preact4b = PreActResNet(channels[4])

        self.preact5a = PreActResNet(channels[4])
        
        self.preact5b = PreActResNet(channels[4])

        self.last = nn.Sequential(
            self.MaxPool,
            nn.BatchNorm2d(channels[4]),
            ReLU,
            nn.Flatten(),
            nn.Linear(channels[4], n_classes)
        )

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
        x = self.conv0(x)
        x = self.preact1(x)
        x = self.conv1(x)
        x = self.preact2a(x)
        x = self.preact2b(x)
        x = self.conv2(x)
        x = self.preact3a(x)
        x = self.preact3b(x)
        x = self.conv3(x)
        x = self.preact4a(x)
        x = self.preact4b(x)
        x = self.MaxPool(x)
        x = self.preact5a(x)
        x = self.preact5b(x)
        out = self.last(x)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
