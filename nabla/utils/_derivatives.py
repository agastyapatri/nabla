"""
Defining the derivatives of the 
activation functions defined in neuron.activations
"""
import numpy as np 

def dSigmoid(x:np.ndarray) -> np.ndarray:
    return np.exp(x) / np.square((1 + np.exp(x)))
    
def dReLU(x:np.ndarray) -> np.ndarray:
    x[x>0] = 1
    x[x<=0] = 0
    return x

def dTanh(x:np.ndarray) -> np.ndarray:
    return np.square((1 / np.cosh(x)))

def dLeakyReLU(x:np.ndarray, slope:float) -> np.ndarray:
    x[x>0] = 1
    x[x<0] = -slope 
    return x 


def dMSE(self, preds:np.ndarray, labels:np.ndarray) -> np.ndarray:
    return  np.mean(np.square(preds-labels))
    

