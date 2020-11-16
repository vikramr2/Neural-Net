import numpy as np
import numpy.linalg as la
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
            #self.layers.append(ReLU(nxt))
        self.layers.append(Softmax(nxt))

        print(" Done")

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            if not isinstance(layer, Softmax):
                x = (x - np.mean(x))/np.std(x)
        self.out = x
        return x

    def xent(self, y):
        return -np.log(self.out[y])

    def backward(self, y):
        upstreams = []
        grad = np.zeros(len(self.out))
        
        for i in range(len(grad)):
            kronecker = 1 if y == i else 0
            grad[i] = self.out[i] - kronecker

        for n in range(len(self.layers) - 1, -1, -1):
            if isinstance(self.layers[n], Linear):
                streams = self.layers[n].backward(grad)
                
                upstreams.append(streams[0])
                grad = streams[1]
            else:
                grad = self.layers[n].backward(grad)

        return (grad, upstreams)
        
