import numpy as np
import scipy as sp
from neuron import activations
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
        input_size:int - the number of features in the input vector == number of neurons in this layer
        output_size:int - the number of output features == the number of neurons in the next layer
        activation:str - the activation function being applied to the outputs of the layer


    """
    def __init__(
            self, 
            input_size:int, 
            output_size:int,
            activation:str,
            dtype=np.float32
            ) -> None:
        self._input = input_size
        self._output = output_size
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size, 1)
        self.activation = activation
        

    def __call__(self, x:np.ndarray) -> np.ndarray:
        z = np.dot(x, self.w) + self.b.T
        return _activation_map[self.activation](z)
    
    def __repr__(self, ) -> str:
        return f"Linear(input_size = {self._input}, output_size = {self._output}, activation={self.activation})"




class MLP:
    """
    Defining a network comprised by Layers
    
    [args]:
        structure:dict - {layer_idx: [[input_size, output_size], activation_function]}
    """
    def __init__(self, structure:dict) -> None:
        self.struct = structure 
        self.net = []
        for i in self.struct:
            [input_size, output_size], activation = self.struct[i]
            self.net.append(Linear(input_size, output_size, activation))

    def __call__(self, x:np.ndarray) -> np.ndarray:
        out = self.net[0](x)
        j = 1 
        while(j < len(self.net)):
            out = self.net[j](out)
            j+=1 
        return out 
    
    def __repr__(self, ) -> str:
        repr = "\n"
        for i in self.net:
            repr += i.__repr__() + "\n"

        return repr




