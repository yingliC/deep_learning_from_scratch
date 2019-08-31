import numpy as np

#ƒ(x0,x1) = x0^2 + x1^2
def function2(x):
    return np.sum(x**2)

# def numercial_gradient(f,x):
#     h = 1e-4
#     grad = np.zeros_like(x)

#     for idx in range(x.size):
#         tmp_val = x[idx]

#         #f(x+h)的计算
#         x[idx] = tmp_val + h
#         fxh1 = f(x)

#         #f(x-h)的计算
#         x[idx] = tmp_val - h
#         fxh2 = f(x)

#         grad[idx] = (fxh1 - fxh2) / (2 *h)
#         x[idx] = tmp_val
    
#     return grad


def numercial_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        #f(x+h)的计算
        fxh1 = f(x)

        #f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 *h)

        x[idx] = tmp_val # 还原值
        it.iternext()
    
    return grad

# print(numercial_gradient(function2,np.array([3.0, 4.0])))
# print(numercial_gradient(function2,np.array([0.0, 2.0])))
# print(numercial_gradient(function2,np.array([3.0, 0.0])))


# f: 要进行最优化的函数
# init_x: 初始值
# lr: 学习率
# step_num: 梯度法的重复次数
# #
def gradient_descent(f, init_x, lr, step_num):
    # x = init_x  此处引用传递，若下面只写一句传递，则最后三个result形同
    x = init_x.copy() # 或者在每局result上面均加上一句赋值

    for i in range(step_num):
        grade = numercial_gradient(f, x)
        x -= lr * grade
    
    return x

# initX = np.array([-3.0, 4.0])
# result1 = gradient_descent(function2, init_x=initX, lr=0.1,step_num=100)
# result2 = gradient_descent(function2, init_x=initX, lr=10,step_num=100)
# result3 = gradient_descent(function2, init_x=initX, lr=1e-10,step_num=100)
# print(result1)
# print(result2)
# print(result3)

