"""
Putting everything together in a Neural Net for approximation

@author Vikram Ramavarapu
"""

import numpy as np
import numpy.linalg as la
from numpy.random import rand
from math import floor
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
            
            if n == len(self.layer_sizes) - 2:
                self.layers.append(ReLU(nxt))
        self.layers.append(Softmax(nxt))

        print(" Done")

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            if not isinstance(layer, Softmax):
                x = (x - x.mean(axis=1)[:, None])/x.std(axis=1)[:, None]
        self.out = x
        return x

    def xent(self, y):
        avg_loss = 0
        count = 0
        for row in self.out:
            y_val = floor(y[count])
            avg_loss -= np.log(row[y_val])
            count += 1
        avg_loss /= len(self.out)
        
        return avg_loss

    def backward(self, y):
        upstreams = []
        grad = np.zeros(self.out.shape)
            
        for i in range(len(y)):
            for j in range(len(grad[i])):
                kronecker = 1 if y[i] == j else 0
                grad[i][j] = self.out[i][j] - kronecker

        for n in range(len(self.layers) - 1, -1, -1):
            if isinstance(self.layers[n], Linear):
                streams = self.layers[n].backward(grad)
                
                upstreams.append(streams[0])
                grad = streams[1]
            else:
                grad = self.layers[n].backward(grad)

        return (grad, upstreams)
        
