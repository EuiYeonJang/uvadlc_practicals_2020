"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        init_mu = 0
        init_sigma = 0.0001

        self.params = {
            "weight": np.random.normal(init_mu, init_sigma, (out_features, in_features)),
            "bias": np.zeros((1, out_features))
        }

        self.grads = {
            "weight": np.zeros((out_features, in_features)), 
            "bias": np.zeros((1, out_features))
        }        
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        out = np.matmul(x, self.params["weight"].T) + self.params["bias"]
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        # gradient wrt weights
        self.grads["weight"] = np.matmul(dout.T, self.x)
        
        # gradient wrt bias
        bias_ones = np.ones((1, dout.shape[0]))
        self.grads["bias"] = np.matmul(bias_ones, dout)

        # gradient wrt input
        dx = np.matmul(dout, self.params["weight"])
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        exp_x = np.exp(x - np.max(x))
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.softmax = out       
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        s, c = dout.shape

        # NOTE: inspiration from https://rockt.github.io/2018/04/30/einsum
        # and https://themaverickmeerkat.com/2019-10-23-Softmax/

        # when i == j
        fst_case = np.einsum("ij,jk->ijk", self.softmax, np.eye(c, c))
        
        # when i != j
        snd_case = np.einsum("ij,ik->ijk", self.softmax, self.softmax)
        
        der_softmax = fst_case - snd_case
        
        dx = np.einsum("ijk,ik->ij", der_softmax, dout)
        #######################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # batch_loss = np.sum(y * np.log(x, where=(x>0)) , axis=1)
        batch_loss = np.sum(y * np.log(x) , axis=1)
        out = (-1/x.shape[0]) * np.sum(batch_loss)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = -(1/x.shape[0]) * np.where(x==0, 0, y / x)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        out = np.where(x>=0, x, np.exp(x)-1)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * np.where(self.x>=0, 1, np.exp(self.x))
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
