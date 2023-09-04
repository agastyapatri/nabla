import numpy as np 
from nabla.tensor import Tensor 

__all__ = [
    "ReLU",
    "LeakyReLU",
    "Tanh",
    "Sigmoid"
]

def ReLU(x:Tensor, dtype=np.float32):
    """
    Rectified Linear Unit
    """
    return Tensor(np.maximum(x.data, np.zeros((x.shape))), requires_grad=True, operation="relu")
    
def LeakyReLU(x:Tensor, slope:float=0.01):
    """
        if x > 0: return x
        else:   return slope*x 
    """
    return Tensor(np.maximum(slope*x.data, x.data), requires_grad=True, operation="leakyrelu")

def Tanh(x:Tensor):
    """
    The Hyperbolic Tan activation function 
    """
    return Tensor(np.tanh(x.data), requires_grad=True, operation="tanh")

def Sigmoid(x:Tensor):
    """
    The sigmoid activation function
    """
    return Tensor(np.exp(x.data)/(1 + np.exp(x.data)), requires_grad=True, operation="sigmoid")

