import numpy as np
import scipy as sp
from nabla.nn import activations
from nabla.tensor import Tensor 

DTYPE = np.float32

_activation_map = {
    "relu" : activations.ReLU,
    "leakyrelu": activations.LeakyReLU,
    "sigmoid": activations.Sigmoid, 
    "tanh": activations.Tanh, 
} 

class Linear:
    """
    Defining a Single Linear Layer
    
    [args]:
        input_size:int - the number of features in the input vector == number of neurons in the previous layer
        output_size:int - the number of output features == the number of neurons in this layer
        activation:str - the activation function being applied to the outputs of the layer

        shape(weight) = (output x input)
        shape(bias) = (output, 1)
        shape(x) = (num_samples, input)
        shape(z) = ()
    """
    def __init__(
            self, 
            input_dim:int, 
            output_dim:int,
            activation:str=None,
            bias:bool=True,
            dtype=np.float32
            ) -> None:
        self._input = input_dim
        self._output = output_dim
        self.weight = Tensor(np.random.randn(output_dim, input_dim), requires_grad=True)
        if bias:
            self.bias = Tensor(np.random.randn(output_dim), requires_grad=True)
        else:
            self.bias = Tensor(np.zeros((output_dim)), requires_grad=True)
        self.activation = activation
        
    def __call__(self, x:np.ndarray) -> np.ndarray:
        return self.forward(x)[1]
    
    def forward(self, x:Tensor):
        z = x@self.weight.transpose() + self.bias
        
        if self.activation == None:
            return z, z 
        else:
            return z, _activation_map[self.activation](z)

    def __repr__(self, ) -> str:
        return f"Linear(input_size = {self._input}, output_size = {self._output}, activation = {self.activation})"
    



