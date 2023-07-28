"""
Loss Functions to be used during the training process
"""

import numpy as np 
import scipy as sp

def MSELoss(preds:np.ndarray, labels:np.ndarray) -> np.ndarray:
    return np.mean(np.square(preds-labels))


