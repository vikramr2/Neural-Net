import numpy as np

class BatchNorm:
    def __init__(self, inputs, eps=np.finfo(np.float32).eps):
        self.in_vector = np.zeros(inputs)
        self.epsilon = eps
        self.beta = np.ones(inputs)
        self.gamma = np.ones(inputs)
        
    def __call__(self, x):
        self.in_vector = x
        
        mu = x.mean(axis=0)
        xmu = x - mu
        
        var = x.var(axis=0)
        sqrtvar = np.sqrt(var + self.epsilon)
        ivar = 1./sqrtvar
        
        xhat = xmu * ivar
        gammax = self.gamma * xhat
        
        self.out = gammax + self.beta
        
        self.cache = (xhat,self.gamma,xmu,ivar,sqrtvar,var)
        
        return self.out
        
    def backward(self, grad):
        xhat,gamma,xmu,ivar,sqrtvar,var = self.cache
        N, D = grad.shape
        dbeta = grad.sum(axis=0)
        dgamma = np.sum(grad*xhat, axis=0)
        dxhat = grad * gamma
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. /(sqrtvar**2) * divar
        dvar = 0.5 * 1. /np.sqrt(var+self.epsilon) * dsqrtvar
        dsq = 1. /N * np.ones((N,D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        dx2 = 1. /N * np.ones((N,D)) * dmu
        dx = dx1 + dx2
        
        self.dx = dx
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx, dgamma, dbeta
        
    def update(self, optimizer):
        self.gamma = optimizer(self.gamma, self.dgamma)
        self.beta = optimizer(self.beta, self.dbeta)
