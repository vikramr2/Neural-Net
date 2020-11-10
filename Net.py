import numpy as np
from numpy.random import rand
from Linear import *
from Functions import *

class Net:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = []

        print("Initializing weights...", end='')

        for n in range(len(self.layer_sizes)-1):
            cur = self.layer_sizes[n]
            nxt = self.layer_sizes[n+1]
            
            self.layers.append(Linear(cur, nxt))
            self.layers.append(ReLU(nxt) if n != len(self.layer_sizes)-2 else Softmax(nxt))

        print(" Done")

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        self.out = x
        return x

    def xent(self, y):
        return -np.log(self.out[y])

    def backward(self, y):
        grad = np.zeros(len(self.out))
        
        for k in range(len(self.out)):
            kronecker = 1 if y == k else 0
            grad[k] = -(kronecker - self.out[y])

        for n in range(len(self.layers) - 2, -1, -1):
            if isinstance(self.layers[n], Linear):
                grad = self.layers[n].backward(grad)[1]
            else:
                grad = self.layers[n].backward(grad)

        return grad
        
