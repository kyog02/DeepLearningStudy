# 誤差逆伝播法を使ったニューラルネットワークで使用する
# 活性化関数レイヤ

import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    # 逆伝播の使い方わからぬな。。
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# relu = Relu()
# x = np.array([[1.0,-0.5],[-2.0,3.0]])
# y = relu.forward(x)
# print(y)
# dx = relu.backward(y)
# print(dx)


