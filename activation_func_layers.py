# 誤差逆伝播法を使ったニューラルネットワークで使用する
# 活性化関数レイヤ
# 活性化関数の逆伝播(backward)は、重み、バイアスの勾配を直接求めない。
# 直接、重み・バイアスの勾配を求めるのは順伝播で重み・バイアスを直接使用するAffineレイヤ


import numpy as np
import activation_func_library
import loss_func_library

class ReluLayer:
    def __init__(self):
        self.mask = None

    # ReLU関数
    # y = max(0,x)
    def forward(self,x):
        # xが0より小さい要素はTrue、xが0より大きい要素はFalseを返す。
        self.mask = (x <= 0)
        out = x.copy()      # 値渡し
        out[self.mask] = 0  # True(0以下)のところを0に書き換える。Falseの箇所はそのまま。
        return out

    # 逆伝播の使い方わからぬな。。
    # ReLU関数の微分
    # dx = if(x > 0){1}
    #      elseif(x <= 0){0}
    def backward(self,dout):
        dout[self.mask] = 0 # 順伝播時のマスクでTrueだったとこを0に置き換える。
        dx = dout
        return dx


class SigmoidLayer:
    def __init__(self):
        self.y = None

    # シグモイド関数
    # y = 1 / (1 + exp(-x))
    def forward(self,x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    # シグモイド関数の微分
    # y : シグモイド関数の出力結果
    # dx = (1 - y) * y
    def backward(self,dout):
        dx = dout * (self.y * (1 - self.y))
        return dx


class AffineLayer:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    # 行列の積
    # y = x * w + b
    # y = np.dot(x,w) + b
    def forward(self,x):
        self.x = x
        return np.dot(x,self.W) + self.b

    # 行列の積の微分
    # dx = dout * W.T
    # dW = x.T * dout
    # db = doutの総和
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx

# 損失関数込みのソフトマックスレイヤ
class SoftmaxWithLossLayer:
    def __init__(self):
        self.loss = None # 損失
        self.y = None    # Softmaxの出力結果
        self.t = None    # 正解ラベル・教師データ

    # ソフトマックス関数
    # y = exp(x) / Σexp(x)
    def forward(self,x,t):
        self.t = t
        self.y = activation_func_library.softmax(x)
        self.loss = loss_func_library.cross_entropy_error(self.y,t)
        return self.loss

    # doutいる？
    # ソフトマックス関数の微分
    # 
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx