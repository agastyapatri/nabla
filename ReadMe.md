#   **NABLA**
_Deep Learning from (almost) scratch_

This project exists to make sure that I understand the core concepts of deep learning: **Gradient Descent** and **Backpropagation**. With time, I intend to implement more sophisticated variants of the Neural Network. 

I intend to write the most performant code I can, but that might take a backseat for haphazard - but understandable - implementations.


## **Notes**

*   **11th August 2023 (Friday)**
    
    ~~1. `nabla.tensor.Tensor` created. This will be the central datatype for all computation.~~
 
    ~~2. `nabla.nn.layers.Linear` needs to be re-written to work with `Tensor` objects.~~

*   **12th August 2023 (Saturday)**
    1. `nabla.nn.templates` needs to be collection of simple, pre-declared neural networks which can be instantiated by declaring `structure`
    2. `nabla` needs to have a `Compose` API similar to `torch.nn.Sequential`. Main functionality to allow for scripting of layers.
    3. `nabla.visualize` is needed; complete functionality needs to be visualization of convergence, training, comp. graphs, etc.
    
    ~~4. `nabla.nn.templates.MLP` needs to work with `Tensor` objects~~

    ~~5. All Activations + Losses need to also work with `Tensor` objects.~~