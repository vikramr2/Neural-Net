import numpy as np
from Linear import *
from Functions import *

x = np.array([1, 2, 3])

linear1 = Linear(3, 4)
reLU = ReLU(4)
linear2 = Linear(4, 5)
softmax = Softmax(5)

x = reLU(linear1(x))
x = softmax(linear2(x))

print(x)
