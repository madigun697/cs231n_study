import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch 
    of N examples, where each example x[i] has shape (d_1, ..., d_k). 
    We will reshape each input into a vector of dimension D = d_1 * ... * d_k, 
    and then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.                                          #
    # You will need to reshape the input into rows.                                                                               #
    ###########################################################################
    # N = x.shape[0]
    # D = np.prod(x.shape[1:])
    # x2 = np.reshape(x, (N, D))
    # out = np.dot(x2, w) + b
    # out = x.reshape(x.shape[0], -1).dot(w) + b
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    #row=N and -1 to automatically reshape into out
    ###########################################################################
    #                             END OF YOUR CODE                                                                                               #
    ###########################################################################
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                              												   #
    ###########################################################################
    N = x.shape[0]
    x_row = x.reshape(N, -1)
    # D = np.prod(x.shape[1:])
    # x2 = np.reshape(x, (N, D))

    dx = dout.dot(w.T).reshape(x.shape)
    dw = (x_row.T).dot(dout)
    db = np.sum(dout, axis=0)
    # db = np.dot(dout.T, np.ones(N))
    ###########################################################################
    #                             END OF YOUR CODE                                                                                                #
    ###########################################################################
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                                                                     #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                                                                                               #
    ###########################################################################
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                                                                  #
    ###########################################################################
    dx = (x >= 0) * dout
    ###########################################################################
    #                             END OF YOUR CODE                                                                                               #
    ###########################################################################
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    ###########################################################################
    # TODO: Implement the softmax function.                                                                                        #
    # You will need to reshape the input into rows.                                                                                #
    ###########################################################################    
    #stable-softmax
    # prob = np.exp(x - np.max(x, axis=1, keepdims=True))
    # prob /= np.sum(prob, axis=1, keepdims=True)

    # exp_score = np.exp(x)
    # prob = exp_score / np.sum(exp_score, axis=1, keepdims=True)
    #
    # N = x.shape[0]
    # # loss = -np.sum(np.log(prob[np.arange(N), y])) / N
    # # print('LOSS %d' %loss)
    # correct_logprobs = -np.log(prob[np.arange(N), y])
    # loss = np.sum(correct_logprobs) / N
    #
    # dx = prob.copy()
    # dx[np.arange(N), y] -= 1
    # dx /= N

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    ###########################################################################
    #                             END OF YOUR CODE                                                                                                #
    ###########################################################################
    return loss, dx
