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
        ) -> list:
    """
    Function to perform backpropagation, given a loss function and 

    1.  Find the delta of the last layer 
    3.  grad(C, b)_layer(last - i) = delta(last_layer - i)
    4.  grad(C, w)_layer(last - i) = delta(last_layer - i)*
    """
    for i in range(len(network)):
        layer = network[-(i+1)]
        

def differentiate() -> None:
    pass 


