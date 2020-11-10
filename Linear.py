import numpy as np
from numpy.random import rand

class Linear:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = rand(inputs, outputs)
        
    def __call__(self, x):
        self.in_vector = np.array(x)
        return x @ self.weights

    def backward(self, grad):
        dx = self.weights.T
        dW = self.in_vector.T

        self.upstream = np.outer(dW, grad)
        self.downstream = grad @ dx

        return (self.upstream, self.downstream)

    def update(self, optimizer):
        self.weights = optimizer(self.weights, self.upstream)
        
        
