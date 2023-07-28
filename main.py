import numpy as np 

from neuron.neuron import Linear, Network

linear = Linear(
    input_size=10,
    output_size=50,
    activation="relu"
)
x = np.random.randn(1, 10)

import timeit
start = timeit.default_timer()
# print(linear(x))
end = timeit.default_timer()
# print(end-start)


