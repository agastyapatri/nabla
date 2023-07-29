import numpy as np 
from neuron.network import Linear, MLP

x = np.random.randn(1, 784) 
structure = {
    0: [[784, 392], "relu"],
    1: [[392, 196], "relu"],
    2: [[196, 98], "relu"],
    3: [[98, 10], "relu"]
}
net = MLP(structure)

for i in range(len(net)):
    print(net[i].w.shape)


# import matplotlib.pyplot as plt 
# def mse1d(x, a):
#     return np.square(x - a)

# x = np.arange(-10, 10, 0.001)
# y = mse1d(x, 0)
# plt.scatter(x, y)
# plt.grid()
# plt.show()
