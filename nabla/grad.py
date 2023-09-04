"""
Module to conduct the learning process. This will include optimization algorithms, backpropagation, training, testing, etc.
"""
import numpy as np 
from .utils import derivative_map
from nabla.tensor import Tensor


def grad(
        x:Tensor
        ) -> None:
    """
    One iteration of parameter updation
    
    [args]:
        network: the model whose parameters are to be updated
        grad_W: the list of the gradients wrt W
        grad_B: the list of the gradients wrt B
    """
    if x.req_grad == False:
        raise Exception(f"requires_grad = False. Grad not valid for leaf nodes!")
    return 




