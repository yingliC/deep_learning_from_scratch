import numpy as np
from collections import OrderedDict
import sys, os
sys.path.append(os.pardir)
from common.util import im2col
from common.layers import Affine, SoftmaxWithLoss, Relu
# from common.functions import relu


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out




class SimpleConvNet:
    def __init__(self, input_dim=(1,28,28), conv_param={'filter_num':30,'fliter_size':5,'pad':0,'stride':1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        # 初始化的最开始部分
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['filter_pad']
        filter_stride = conv_param['filter_stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size/2))

        #权重参数的初始化部分
        self.param = {}
        self.param['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.param['b1'] = np.zeros(filter_num)
        self.param['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.param['b2'] = np.zeros(hidden_size)
        self.param['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.param['b3'] = np.zeros(output_size)

        # 生成必要层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.param['W1'], self.param['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['pool'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.param['W2'], self.param['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.param['W3'], self.param['b3'])
        self.last_layer = SoftmaxWithLoss()
        
