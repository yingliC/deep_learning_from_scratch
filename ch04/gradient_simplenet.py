import sys, os
sys.path.append(os.pardir)

import numpy as np
from gradient import numercial_gradient
from common.functions import cross_entropy_error, softmax

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)  #用高斯分布进行初始化
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print("net.W:")
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print("p:")
print(p)
print("np.argmax(p):")
print(np.argmax(p))

t = np.array([0, 0, 1])
print("net.loss(x,t):")
print(net.loss(x, t))