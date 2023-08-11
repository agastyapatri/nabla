import numpy as np 
from nabla import Tensor 
from nabla.nn.templates import MLP
from nabla.nn.layers import Linear



np.random.seed(0)
DTYPE = np.float32 

x = np.random.randn(10, 20)
y = np.ones((10, 1))

ytensor = Tensor(y)
xtensor = Tensor(x)

structure = {
    0:  [[20, 20], "relu"],
    1:  [[20, 1], "relu"],
}
net = MLP(structure)

print(net(xtensor).shape)


# weight = Tensor(np.random.randn(40, 20))

# print(weight.transpose().shape)
