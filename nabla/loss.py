"""
Loss Functions to be used during the training process
"""
import numpy as np 

class MSELoss:

    def __call__(
            self, 
            x:np.ndarray, 
            y:np.ndarray
            ) -> np.ndarray:
        return np.mean(np.square(x-y))
    
    def grad(
            self, 
            x:np.ndarray, 
            y:np.ndarray
            ) -> np.ndarray:
        """
        derivative of MSE with respect to x
        """
        return 2*np.mean(x - y)


class CELoss:
    def __init__(self, ) -> None:
        pass

    def __call__(
            self, 
            x:np.ndarray, 
            y:np.ndarray
            ) -> None:
        return x 
    
    def grad(self, x:np.ndarray, y:np.ndarray):
        pass 


    
