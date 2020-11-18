import numpy as np

class SGD:
    def __init__(self, lr, decay=0):
        self.lr = lr
        self.iter = 0
        self.rate = decay

    def __call__(self, weights, upstream):
        curr_lr = self.lr * np.exp(-self.rate*self.iter)

        return weights - curr_lr * upstream

    def advance(self):
        self.iter += 1

class Adam:
    
