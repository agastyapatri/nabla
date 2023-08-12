import numpy as np 
from nabla import Tensor 
from nabla.nn.templates import MLP
from nabla.nn.layers import Linear
from nabla.utils import structures
np.random.seed(0)
DTYPE = np.float32 


x = np.random.randn(100, 10)
y = np.ones((100, 1))


"""
    xtensor has 100 samples, 10 features per sample
    ytensor has 100 samples, 1 feature per sample
"""
xtensor = Tensor(x)
ytensor = Tensor(y)

structure = {
    0:  [[10, 20], "sigmoid"],
    1:  [[20, 40], "sigmoid"],
    2:  [[40, 80], "sigmoid"],
    3:  [[80, 40], "sigmoid"],
    4:  [[40, 1], "relu"],
}
net = MLP(structure)

for weight in net.W:
    print(isinstance(weight, Tensor))