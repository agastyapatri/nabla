#   **Neural Networks in NumPy**
_Deep Learning from (almost) scratch_

This project exists to make sure that I understand the core concepts of deep learning: **Gradient Descent** and **Backpropagation**. With time, I intend to implement more complicated variants of the Neural Network. 

I intend to write the most performant code I can, but that might take a backseat for haphazard - but understandable - implementations.


## **Notes**

*   **11th August 2023 (Friday)**
    
    1. `nabla.tensor.Tensor` created. This will be the central datatype for all computation.
    2. `nabla.nn.layers.Linear` needs to be re-written to work with `Tensor` objects. 
    3. `nabla.nn.templates` needs to be collection of simple, pre-declared neural networks which can be instantiated by declaring `structure`
    4. `nabla` needs to have a `Compose` API similar to `torch.nn.Sequential`. Main functionality to allow for scripting of layers.
    5. `nabla.visualize` is needed; complete functionality needs to be visualization of convergence, training, comp. graphs, etc.1 
