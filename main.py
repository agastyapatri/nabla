import numpy as np 
from neuron.network import Linear, MLP
from nabla.loss import Loss
from nabla import backward



x = np.random.randn(1, 784) 
structure = {
    0: [[784, 392], "relu"],
    1: [[392, 196], "relu"],
    2: [[196, 98], "relu"],
    3: [[98, 10], "relu"]
}
net = MLP(structure)

loss_fun = Loss.MSELoss

backward(net, None, x)