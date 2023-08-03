"""
Module to conduct the learning process. This will include optimization algorithms, backpropagation, training, testing, etc.
"""
from nabla.utils import _derivatives
from neuron.activations import *
from neuron import network 
from nabla.loss import MSELoss

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
    """
    Z = [] 
    A = [] 
    Z.append(network[0](x)[0])
    A.append(network[0](x)[1])
    
    #   forward pass; finding the outputs and activations at each layer  
    for i in range(1, len(network)):
        z, a = network[i](A[i-1])
        A.append(a)
        Z.append(z)

    #   backward pas
    grad_W = []
    grad_B = []
    loss_grad = loss.grad(network(x), y)

    #   delta is the error in the output layer
    delta = _derivative_map[network[-1].activation](Z[-1])*loss_grad 
    
    #   iterating through the reversed network  
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
        grad_B:list
        ) -> None:
    """
    Function to perform gradient descent using the gradient values of the loss function with respect to the parameters
    
    [args]:
        network: the model whose parameters are to be updated
        grad_W: the list of the gradients wrt W
        grad_B: the list of the gradients wrt B
        
    """
    print(len(network))
    print(len(grad_W))
    print(len(grad_B))

