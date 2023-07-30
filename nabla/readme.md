## **Backpropagation** 
is an algorithm used to compute gradients, most importantly of the cost function.

### **Introduction**

At the heart of backpropagation is an expression for the partial derivative $\frac{\partial C}{\partial x}$ of the cost function with respect to any weight or bias of the network. 
The expression tells us how quickly the Cost function changes when the parameters are changed. 

*   $w_{jk}^{l}$ is the weight from the $k^{th}$ neuron of the $l-1^{th}$ layer to the $j^{th}$ neuron of the $l^{th}$ layer
*   $b_{j}^{l}$ is the bias of the $j^{th}$ neuron in the $l^{th}$ layer.
*   $a_{j}^{l}$ is the activation of the $j^{th}$ neuron in the $l^{th}$ layer.
*   With the above notations, we can express activations as 
    $$a_{j}^{l} = \sigma(\Sigma_{i}^{n}w_{ji}^{l}a_{i}^{l-1} + b_{j}^{l})$$
    Where n is the number of neurons in the (l-1)th layer
*   In Vecotrized Form, 
    $$a^{l} = \sigma(w^la^{l-1} + b^l)$$

### **Two Assumptions about the Cost Function**
The goal of backpropagation is to compute $\frac{\partial C}{\partial w}$ and $\frac{\partial C}{\partial b}$. 


*   The first assumption is that the cost function can be written as an average $C   = \frac{1}{n}\Sigma_{x}C_x$ over cost functions $C_x$ for individual training examples $x$.
*   The cost function must be able to be represented as a function of the outputs from the neural network 

    $C = F(a^{L})$ where L = number of layers in the network

*   If $C = \frac{1}{n}||y - a^{L}||^2$

### **The Four Fundamental Equations behind backpropagation**
We need $\frac{\partial C}{\partial w^{l}_{jk}},\frac{\partial C}{\partial b^{l}_{j}}$.
For this, we need a quantity known as the error $\delta_{j}^{l}$ in the jth error in the lth layer. This error is then related to the two quantities we need. 

The BP Equations are:  

1.  
    $$\delta^{L}_{j} = \frac{\partial C}{\partial a_{j}^{L}} \frac{\partial a^{L}_{j}}{\partial z^{L}_{j}}$$  
    $$\delta^{L} = \nabla_{a}C * \sigma'(z^{L})$$

2.  $$\delta^{l} = ((w^{l+1})^{T} \delta^{l+1}) * \sigma'(z^{l})$$

3.  $$\delta_{j}^{l} = \frac{\partial C}{\partial b_{j}^{l}}$$

4.   $$a_{l-1}^{k}\delta_{j}^{l} = \frac{\partial C}{\partial w_{jk}^{l}}$$


