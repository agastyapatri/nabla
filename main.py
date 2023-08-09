import numpy as np 
from nabla import Neuron 
from neuron.activations import *
from neuron import Linear, MLP
from nabla.loss import MSELoss
np.random.seed(0)
import torch 
import torch.nn as nn 



x = np.random.randn(100, 10)
layer_nabla = Linear(input_dim=10, output_dim=20)


struct = {
    0   : [[10, 20], "relu"],
    1   : [[20, 40], "relu"],
    2   : [[40, 80], "relu"],
    3   : [[80, 40], "relu"],
    4   : [[40, 20], "relu"],
    5   : [[20, 10], "relu"],
}


net = MLP(structure=struct)

import timeit
start = timeit.default_timer()
net(x)
end = timeit.default_timer()
print(end-start)