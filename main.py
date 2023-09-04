import numpy as np 
np.random.seed(0)
DTYPE = np.float32 

from nabla import Tensor 
from nabla.nn.templates import MLP
# from nabla import backprop
from nabla.nn.layers import Linear

x = np.random.randn(100, 784)
y = np.ones((100, 10))


"""
    xtensor has 100 samples, 10 features per sample
    ytensor has 100 samples, 1 feature per sample
"""
xtensor = Tensor(x)
ytensor = Tensor(y)

# Defning the Multilayer Perceptron
structure = {
    0:  [[784, 392], "relu"],
    1:  [[392, 196], "relu"],
    2:  [[196, 10], "relu"],
}
net = MLP(structure)

def backprop(x):
    """
    Performing simple backpropagation 

    1.  forward pass; collect z's and a's for the 
    """
    Z = []
    A = []
    
    #   forward pass; collecting activations and weighted sums
    i = 0 
    z = x
    while i<len(net):
        z, a = net[i].forward(z)
        Z.append(z)
        A.append(a)
        i+=1
    


backprop(xtensor)

