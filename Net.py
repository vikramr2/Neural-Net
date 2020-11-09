import numpy as np
from numpy.random import rand

class Net:
    def __init__(self, num_inputs, num_outputs):
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.middle_layers = []
        self.biases = []
        self.weights = []

    def add_layer(self, num_nodes):
        self.middle_layers.append(num_nodes)

    def initialize(self):
        print("Initializing weights and biases...", end='')

        # reset biases and weights
        self.biases = []
        self.weights = []

        # append all middle layers
        for i in range(len(self.middle_layers)):
            curr = self.middle_layers[i]
            prev = self.inputs if i == 0 else self.middle_layers[i-1]
            
            self.biases.append(np.zeros(curr))
            self.weights.append(rand(prev, curr))

        # append final layers
        self.weights.append(rand((self.inputs if len(self.middle_layers) == 0
                                     else self.middle_layers[-1]), self.outputs))
        self.biases.append(np.zeros(self.outputs))

        print(" Done")

        
