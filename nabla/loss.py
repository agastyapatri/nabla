"""
Loss Functions to be used during the training process
"""
import numpy as np 
from .tensor import Tensor 


class MSELoss:
    """
    Mean Squared Error Loss 
    """
    def __call__(
            self, 
            x:Tensor, 
            y:Tensor
            ) -> np.ndarray:
        return Tensor(np.mean(np.square(x.data-y.data)), requires_grad=True, operation="mse")
    
    def grad(
            self, 
            x:np.ndarray, 
            y:np.ndarray
            ) -> np.ndarray:
        """
        derivative of MSE with respect to x
        """
        return Tensor(2*np.mean(x.data - y.data), requires_grad=True, operation="dmse")


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

    
