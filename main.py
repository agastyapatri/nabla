import numpy as np 
np.random.seed(0)
DTYPE = np.float32 

from nabla.tensor import Tensor 
x = np.random.randn(10, 20)
y = np.ones((20, 1))

ytensor = Tensor(y)
xtensor = Tensor(x)

print(xtensor.reshape((20,10)).shape)