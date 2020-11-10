import numpy as np

class ReLU:
    def __init__(self, inputs):
        self.in_vector = np.zeros(inputs)
        
    def __call__(self, x):
        self.in_vector = x
        return np.maximum(0, x)

    def backward(self, grad):
        x = self.in_vector
        
        x[x<=0] = 0
        x[x>0] = 1

        return grad @ np.diag(x)

class Softmax:
    def __init__(self, inputs):
        self.in_vector = np.zeros(inputs)

    def __call__(self, x):
        self.in_vector = x
        
        exps = np.exp(x)
        x = exps / np.sum(exps)

        return x
