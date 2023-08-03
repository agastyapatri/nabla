import numpy as np 
import nabla 
from neuron.network import Linear, MLP
from nabla.loss import MSELoss
from nabla.grad import backprop, descent


x = np.random.randn(1, 784) 
y = np.random.randn(1, 10)
structure = {
    0: [[784, 392], "relu"],
    1: [[392, 196], "relu"],
    2: [[196, 98], "relu"],
    3: [[98, 10], "relu"]
}
net = MLP(structure)
loss_fun = MSELoss()
grad_w, grad_b = backprop(network=net, loss=loss_fun, x=x, y=y)
descent(net, grad_w, grad_b)




