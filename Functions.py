import numpy as np

class ReLU:
    def __init__(self, inputs):
        self.in_vector = np.zeros(inputs)
        
    def __call__(self, x):
        # RELU(x) = max(0, x)
        
        # save input vector to use for backprop
        self.in_vector = x
        return np.maximum(0, x)

    def backward(self, grad):
        # gradient processes input vector
        x = self.in_vector
        
        # if entries are less than or equal to 0, set 0, else set 1
        x[x<=0] = 0
        x[x>0] = 1

        # entrywise multiply to finish baclkwards gradient
        return np.multiply(grad, x)

class Softmax:
    def __init__(self, inputs):
        self.in_vector = np.zeros(inputs)

    def __call__(self, x):
        # store input vector for backprop
        self.in_vector = x
        
        # Softmax(exp(x)/sum{exp(x)}
        exps = np.exp(x)
        x = exps / exps.sum(axis=1)[:,None]
        
        # store output for backprop
        self.out = x
        return x
        
    def backward(self, grad):
        n = len(self.out[0])
        J = np.zeros((n, n))
        
        # build softmax jacobian for each entry
        for k in range(len(grad)):
            for i in range(n):
                for j in range(n):
                    kronecker = 1 if i == j else 0
                    J[i][j] = self.out[k][i]*(kronecker - self.out[k][j])
                    
            # multiply each row by respective jacobian
            grad[k] = J @ grad[k]
            
        return grad


