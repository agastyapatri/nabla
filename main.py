import numpy as np 
from neuron.network import Linear, MLP
from nabla.utils.loss import Loss

x = np.random.randn(1, 784) 
structure = {
    0: [[784, 392], "relu"],
    1: [[392, 196], "relu"],
    2: [[196, 98], "relu"],
    3: [[98, 10], "relu"]
}
net = MLP(structure)

loss_fun = Loss.MSELoss


def backpropagation(network:MLP, loss:Loss) -> None:
    """
    Function to perform backpropagation, given a loss function and 
    """
    num_layers = len(network)
    for i in range(len(network)):
        layer = network[-(i+1)]

        #-----------------------------------------------------------------------#
        # The Backpropagation Algorithm
        # #
        #-----------------------------------------------------------------------#

backpropagation(net, loss_fun)


    
    
