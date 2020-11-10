import numpy as np

class SGD:
    def __init__(self, lr, decay=0, momentum=0, friction=0.5):
        self.lr = lr
        self.epoch = 0
        self.rate = decay
        self.m = momentum
        self.beta = friction

    def __call__(self, weights, upstream):
        curr_lr = self.lr * np.exp(-self.rate*self.epoch)
        
        M = np.full(upstream.shape, self.beta * self.m)
        M = M - curr_lr * upstream

        return weights + M

    def advance(self):
        self.epoch += 1
