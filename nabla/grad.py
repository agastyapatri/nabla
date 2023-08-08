"""
Module to conduct the learning process. This will include optimization algorithms, backpropagation, training, testing, etc.
"""
from nabla.utils import _derivatives
from neuron.activations import *
from neuron import network 
from nabla.loss import MSELoss
import numpy as np 

_derivative_map = {
    "relu": _derivatives.dReLU, 
    "sigmoid": _derivatives.dSigmoid,
    "leakyrelu": _derivatives.dLeakyReLU,
    "tanh": _derivatives.dTanh,
}


def backprop(
        network:network.MLP, 
        loss:MSELoss, 
        x:np.ndarray,
        y:np.ndarray,
        ) -> tuple:
    """
    Performing Gradient Descent on a given network 

    [args]:
        network:    The network whose parameters need to be updated
        loss:   The Loss function to be differentiated
        x:  Input
    """
    Z = [network[0](x)[0]] 
    A = [network[0](x)[1]] 
    
    #   forward pass; finding the outputs and activations at each layer  
    for i in range(1, len(network)):
        z, a = network[i](A[i-1])
        A.append(a)
        Z.append(z)

    #   delta is the error in the output layer
    loss_grad = loss.grad(network(x), y)
    delta = _derivative_map[network[-1].activation](Z[-1])*loss_grad 
    grad_W = [delta*A[-1]]
    grad_B = [delta]
    
    #   backward pass; finding the gradients of all the layers starting from the output layer.
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

    return grad_W, grad_B



def descent(
        network:network.MLP, 
        grad_W:list, 
        grad_B:list,
        learning_rate:float
        ) -> None:
    """
    One iteration of parameter updation
    
    [args]:
        network: the model whose parameters are to be updated
        grad_W: the list of the gradients wrt W
        grad_B: the list of the gradients wrt B
    """
    import copy 
    new_net = copy.deepcopy(network)
    new_w, new_b = [], []
    for i in range(len(new_net)):
        w, b = new_net.W[i], new_net.B[i]
        dw, db = grad_W[-(i+1)], grad_B[-(i+1)]
        new_w.append(w - learning_rate*dw)
        new_b.append(b - learning_rate*db)
    new_net.W = new_w
    new_net.B = new_b
    return new_net




