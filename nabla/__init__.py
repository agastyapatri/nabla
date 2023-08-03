"""
Module to conduct the learning process. This will include optimization algorithms, backpropagation, training, testing, etc.
"""
from nabla.utils import _derivatives
from neuron.activations import *
from neuron import network 
from nabla.loss import Loss


_derivative_map = {
    "relu": _derivatives.dReLU, 
    "sigmoid": _derivatives.dSigmoid,
    "leakyrelu": _derivatives.dLeakyReLU,
    "tanh": _derivatives.dTanh,
    "mse":  _derivatives.dMSE 
}


def backward(
        network:network.MLP, 
        loss:Loss, 
        x:np.ndarray,
        y:np.ndarray,

        ) -> list:
    """
    Function to perform backpropagation, given a loss function and 

    1.  Find the delta of the last layer 
    3.  grad(C, b)_layer(last - i) = delta(last_layer - i)
    4.  grad(C, w)_layer(last - i) = delta(last_layer - i)*
    """
    Z = [] 
    A = [] 
    Z.append(network[0](x)[0])
    A.append(network[0](x)[1])
    
    #   forward pass  
    for i in range(1, len(network)):
        z, a = network[i](A[i-1])
        A.append(a)
        Z.append(z)

    #   backward pass
    grad_W = []
    grad_B = []
    # deltas = []
    # for i in range(len(network)-1, -1, -1):
    #     deriv = _derivative_map[network[i].activation]
    #     if i == len(network)-1:
    #         delta = deriv(Z[i])*(A[i] - y)
    #     else:
    #         delta = deriv(Z[i])*(network.W[i+1] * delta)
    #     print(delta.shape)
    #     grad_B[i] = delta
    #     grad_W[i] = A[i-1]*delta    

    #-----------------------------------------------------------------------#
    # Algorithm for performing backpropagation on the network
    # 1.    calculate the deltas at the output layer
    # 2.    #
    #-----------------------------------------------------------------------#
    delta = _derivative_map[network[-1].activation](Z[-1])*(A[-1] - y)
    i = 1 
    while i < len(network):
        layer = network[-(i+1)]
        z = Z[-(i+1)]
        a = A[-(i+1)]
        dsigma = _derivative_map[layer.activation](z) 
        delta = dsigma * (np.dot(delta, network.W[-i].T))
        grad_B.append(delta)
        grad_W.append(a*delta)
        i += 1

    print(len(grad_B), len(grad_W))