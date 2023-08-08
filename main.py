import numpy as np 
from nabla import Neuron 
from neuron.activations import *
from neuron import Linear 
from nabla.loss import MSELoss
np.random.seed(0)


input_features = 10
output_features = 20 
num_samples = 100



x = np.random.randn(num_samples, input_features)
w = np.random.randn(output_features, input_features)
b = np.random.randn(1, output_features)

a1 = Sigmoid(np.dot(x, w.T) + b)

# print(w.shape)
# print(b.shape)

print(a1.shape)



import torch 
import torch.nn as nn 


layer = nn.Linear(10, 20)
x = torch.randn(100, 10)
# print(layer.weight.shape)
# print(layer.bias.shape)
print(layer(x).shape)