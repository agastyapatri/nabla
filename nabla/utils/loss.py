"""
Loss Functions to be used during the training process
"""

import numpy as np 
import scipy as sp

def MSELoss(preds:np.ndarray, labels:np.ndarray) -> np.ndarray:
    return np.mean(np.square(preds-labels))


def CELoss(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """
    The Cross Entropy Loss between two distributions.
    
    [args]:
        x:  distributoin 1
        y:  distributoin 2
        
    """
    pass



    
