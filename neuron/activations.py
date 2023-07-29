import numpy as np 
import scipy as sp 

def ReLU(x:np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Rectified Linear Unit
    """
    return np.maximum(x, np.zeros((x.shape)))
    


def LeakyReLU(x:np.ndarray, slope:float=0.01) -> np.ndarray:
    """
    Leaky Rectified Linear Unit
        
    [args]: 
    x:float - the input
    slope:float - the negative slope of the line if x< 0 
    """
    comp = np.zeros((x.shape))
    return comp 


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
