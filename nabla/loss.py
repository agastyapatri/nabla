"""
Loss Functions to be used during the training process
"""
import numpy as np 

class MSELoss:
    """
    Mean Squared Error Loss 
    """
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
    """
    Cross Entropy Loss 
    """
    def __call__(
            self, 
            a:np.ndarray, 
            y:np.ndarray
            ) -> np.ndarray:
        return -1*np.mean(y*np.log(a) + (1-y)*np.log(1-a)) 
    
    def grad(self, x:np.ndarray, y:np.ndarray):

        pass 

    
