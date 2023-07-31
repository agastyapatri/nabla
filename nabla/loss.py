"""
Loss Functions to be used during the training process
"""
import numpy as np 

class Loss:
    def __init__(self, option:str) -> None:
        self.option = option
        self._loss_map = {
            "mse" : self.MSELoss,
            "cel" : self.CELoss,
        }

    def MSELoss(self, preds:np.ndarray, labels:np.ndarray) -> np.ndarray:
        return np.mean(np.square(preds-labels))


    def CELoss(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        The Cross Entropy Loss between two distributions.

        [args]:
            x:  distributoin 1
            y:  distributoin 2

        """
        pass
    
    def __call__(self, x:np.ndarray, y:np.ndarray):
        return self._loss_map[self.option.lower()](x, y)
    
    
    def backward(self):
        pass 
    


    
if __name__ == "__main__":
    loss_fn = Loss("mse")
    x = np.random.randn(100, 100)
    y = np.random.randn(100, 100)
    import timeit
    start = timeit.default_timer()

    loss_fn(x, y)
    end = timeit.default_timer()
    print(end-start)

