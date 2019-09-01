import numpy as np

def smooth_curve(x):
    """用于使损失函数的图形变圆滑"""
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')

    return y[5:len(y)-5]