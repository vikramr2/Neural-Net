import numpy as np
from numpy.random import rand

class Linear:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.array([x - 0.5 for x in rand(inputs, outputs)])*2
        
    def __call__(self, x):
        self.in_vector = np.array(x)
        return x @ self.weights

    def backward(self, grad):
        # UPSTREAM: gradient with respect to weight matrix
        # DOWNSTREAM: gradient with respect to output vector, used for backprop
    
        dx = self.weights.T
        dW = self.in_vector.T
        
        # batch gradient is 1/n of gradient sum
        self.upstream = (dW @ grad)/len(self.in_vector)
        self.downstream = grad @ dx

        return (self.upstream, self.downstream)

    def update(self, optimizer):
        # using SGD or Adam, update weights
        self.weights = optimizer(self.weights, self.upstream)
        
        
