import numpy as np 
np.random.seed(0)

class Tensor:
    """
    The Central data structure of NABLA. This is built on top of np.ndarrays to enable computational graphs and autodiff.
    Tensor is a class that wraps around numpy arrays. There is no way to auto-populate Tensors without calling numpy.

    [args]:
        data:   the np.ndarray underlying the tensor
        requires_grad: True if not a leaf tensor else False
        operation:  the operation which comprises the Tensor
    """
    def __init__(
            self, 
            data:np.ndarray,
            requires_grad:bool=False, 
            operation:str=None,
            ) -> None:
        self.data = data 
        self.req_grad = requires_grad
        self.op = operation
        self.dtype = self.data.dtype 
        self.shape = self.data.shape 


    #   Arithmetic Operations
    def __add__(self, other):
        return Tensor(self.data + other.data, requires_grad=True, operation="add")
    
    def __sub__(self, other):
        return Tensor(self.data - other.data, requires_grad=True, operation="sub")
    
    def __mul__(self, other):
        return Tensor(self.data * other.data, requires_grad=True, operation="mul")
    
    def __matmul__(self, other):
        return Tensor(self.data @ other.data, requires_grad=True, operation="matmul") 
    
    def __pow__(self, x):
        return Tensor(self.data ** x, requires_grad=True, operation="power")

    #   other operations
    def mean(self, axis):
        return Tensor(np.mean(self.data, axis=axis))    
  
    def std(self, axis):
        return Tensor(np.std(self.data, axis=axis))    
    
    def max(self, axis):
        return Tensor(np.max(self.data, axis=axis))    
    
    def sqrt(self,):
        return Tensor(np.sqrt(self.data), requires_grad=self.req_grad)    
    
    #   transcendental functions
    def exp(self, ):
        return Tensor(np.exp(self.data), requires_grad=self.req_grad)
    
    def log(self, ):
        return Tensor(np.log(self.data), requires_grad=self.req_grad)
    
    def sin(self, ):
        return Tensor(np.sin(self.data), requires_grad=self.req_grad)
    
    def cos(self, ):
        return Tensor(np.cos(self.data), requires_grad=self.req_grad)

    def tanh(self, ):
        return Tensor(np.tanh(self.data), requires_grad=self.req_grad)

    def reshape(self, size:tuple):
        return Tensor(np.reshape(self.data, size), requires_grad=self.req_grad)
    
    def transpose(self, ):
        return Tensor(self.data.T, requires_grad=self.req_grad)

    #   Metadata
    def __repr__(self, ) -> str:
        repr = "Tensor(\n"
        repr += str(self.data)
        repr += f",\nrequires_grad={self.req_grad}, "
        repr += f" operation = {self.op}, "
        repr += f" dtype = {self.dtype}"
        repr += "\n)"
        return repr 
    
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self, ) -> int:
        return len(self.data)



if __name__ == "__main__":
    x = np.random.randn(10, 10)
    xtensor = Tensor(x)
    print(xtensor.shape)
