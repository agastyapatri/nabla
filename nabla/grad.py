"""
Backpropagation of gradients

Backpropagation is used to effectively train a neural network 
via the chain rule. 
In simple terms, after each forward pass through a network, backprop 
performs a backward pass while adjusting the model's parameters.

Backpropagation adjusts the weights of the connections the network so as to 
minimize a measure of the difference between the actual output vector of the 
network and the desired output vector.
"""
from nabla.utils import _derivatives

_derivative_map = {
    "relu": _derivatives.dReLU, 
    "sigmoid": _derivatives.dSigmoid,
    "leakyrelu": _derivatives.dLeakyReLU,
    "tanh": _derivatives.dTanh,
    "mse":  _derivatives.dMSE 
}


def backward(network, loss) -> None:
    """
    Function to perform backpropagation, given a loss function and 

    [args]:
    """

    pass


def differentiate() -> None:
    pass 

