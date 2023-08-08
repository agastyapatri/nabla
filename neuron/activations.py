import numpy as np 

__all__ = [
    "ReLU",
    "LeakyReLU",
    "Tanh",
    "Sigmoid"
]



def ReLU(x:np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Rectified Linear Unit
    """
    return np.maximum(x, np.zeros((x.shape)))
    

def LeakyReLU(x:np.ndarray, slope:float=0.01) -> np.ndarray:
    """
        if x > 0: return x
        else:   return slope*x 
    """
    return np.maximum(slope*x, x)


def Tanh(x:np.ndarray) -> np.ndarray:
    """
    The Hyperbolic Tan activation function 
    """
    return np.tanh(x)

def Sigmoid(x:np.ndarray) -> np.ndarray:
    """
    The sigmoid activation function
    """
    return np.exp(x)/(1 + np.exp(x))

