"""
Pre-built networks that only need to be given a structure to be properly defined. 

Includes:

    1. Multi Layer Perceptron

To be added:
    2. Convolutional Neural Networks
    3. Recurrent Neural Networks
"""
import numpy as np 
from .layers.linear import Linear 


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
