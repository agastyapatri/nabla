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
            dtype=np.float32
            ) -> None:
        self._input = input_dim
        self._output = output_dim
        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim)
        self.activation = activation
        
    def __call__(self, x:np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weight.T) + self.bias 
        if self.activation == None:
            return z 
        else:
            return _activation_map[self.activation](z)
    
    def __repr__(self, ) -> str:
        return f"Linear(input_size = {self._input}, output_size = {self._output}, activation={self.activation})"
    

class MLP:
    """
    Defining a Multi Layer Perceptron comprised by Linear Layers. 
    
    [args]:
        structure:dict - {layer_idx: [[input_size, output_size], activation_function]}
    """
    def __init__(self, structure:dict) -> None:
        self.struct = structure 
        self.net = []
        for i in self.struct:
            [input_size, output_size], activation = self.struct[i]
            self.net.append(Linear(input_size, output_size, activation))
        self.W = [self.net[i].weight for i in range(len(self.net))]
        self.B = [self.net[i].bias for i in range(len(self.net))]

    def __call__(self, x:np.ndarray) -> tuple:
        out = self.net[0](x)
        j = 1 
        while(j < len(self.net)):
            out = self.net[j](out)
            j+=1 
        return out 
    
    def __repr__(self, ) -> str:
        repr = "\n"
        repr += "MultiLayerPerceptron(\n"
        for i in self.net:
            repr += "   " + i.__repr__() + "\n"
        repr += ")\n"

        return repr

    def __getitem__(self, i:int) -> Linear:
        return self.net[i]

    def __len__(self, ) -> int:
        return len(self.net)


