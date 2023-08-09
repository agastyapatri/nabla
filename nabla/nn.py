import numpy as np 
class Neuron:
    """
    Defining a Single Neuron.
    
    [args]:
        activations:    the nonlinearity being applied to the weighted sum
    """
    def __init__(self, activation:str) -> None:
        self.activation = activation 

    def __call__(self, x) -> np.ndarray:
        w = np.random.randn(x.shape[0], 1)
        b = np.random.randn(1, 1)
        z = np.dot(w.T, x) + b 
        a = self.activation(a)
        return a 







# class Module:
#     """
#     Base class for neural netwowrks. A copy of PyTorch's nn.Module
#     """
#     def __init__(self, ) -> None:
#         pass
    
#     def __call__(self, x) -> np.ndarray:
#         return self.forward(x)

#     def forward(self, x) -> np.ndarray:
#         return None
    
