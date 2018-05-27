import numpy as np

from layers import *
from layer_utils import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary 
    and will be learned using the Solver class.
    """

    def __init__(self,
                 hidden_dims,
                 input_dims=3 * 32 * 32,
                 num_classes=10,
                 lambda_reg=0.0,
                 weight_scale=1e-2,
                 dtype=np.float32):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - lambda_reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.lambda_reg = lambda_reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in  									     #
        # the self.params dictionary. Store weights and biases for the first layer                                      #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be                             #
        # initialized from a normal distribution centered at 0 with standard                                             #
        # deviation equal to weight_scale. Biases should be initialized to zero.                                        #
        ############################################################################
        # For dictionary format,
        #    - params['W_i'] = ~ 
        #    - params['b_i'] = ~~ 
        dimensions = [input_dims] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            self.params['W_%d' % (i + 1)] = np.random.normal(0, weight_scale, (dimensions[i], dimensions[i + 1]))
            self.params['b_%d' % (i + 1)] = np.zeros(dimensions[i + 1])

        ############################################################################
        #                             END OF YOUR CODE                                                                                                  #
        ############################################################################

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
                 names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing 							 #
        # the class scores for X and storing them in the scores variable.          										     #
        ############################################################################

        X = np.reshape(X, [X.shape[0], -1])

        layers = {}
        layers[0] = X
        cache_layer = {} # fc_cache, relu_cache

        for i in range(1, self.num_layers):
            W = self.params['W_%d' % i]
            b = self.params['b_%d' % i]

            layers[i], cache_layer[i] = affine_relu_forward(layers[i - 1], W, b)

        W_layers = 'W_%d' % self.num_layers
        b_layers = 'b_%d' % self.num_layers
        scores, cache_scores = affine_forward(layers[self.num_layers-1], self.params[W_layers], self.params[b_layers])
        ############################################################################
        #                             END OF YOUR CODE                            																		 #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the 							 #
        # loss in the loss variable and gradients in the grads dictionary. Compute									 #
        # data loss using softmax, and make sure that grads[k] holds the gradients								 #
        # for self.params[k]. Don't forget to add L2 regularization!               											     #                                                                       
        #                                                                          																						 #
        # NOTE: To ensure that your implementation matches ours and you pass the                             #
        # automated tests, make sure that your L2 regularization includes a factor                                 #
        # of 0.5 to simplify the expression for the gradient.                      													 #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        loss += 0.5 * self.lambda_reg * (np.sum(np.square(self.params['W_%d' % self.num_layers])))

        dx = [None] * (self.num_layers + 1)

        dx[self.num_layers], grads[W_layers], grads[b_layers] = affine_backward(dscores, cache_scores)
        grads[W_layers] += -self.lambda_reg * self.params[W_layers]

        for i in reversed(range(1, self.num_layers)):
            dx[i], grads['W_%d' % i], grads['b_%d' % i] = affine_relu_backward(dx[i+1], cache_layer[i])
            grads['W_%d' % i] += -self.lambda_reg * self.params['W_%d' % i]

        loss += 0.5 * self.lambda_reg * (np.sum(np.square(self.params['W_%d' % i])))

        ############################################################################
        #                             END OF YOUR CODE                                                                                                  #
        ############################################################################

        return loss, grads
