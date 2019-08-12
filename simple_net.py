# ライブラリ
import numpy as np

# 自作ファイル
import activation_func_library
import loss_func_library

# 勾配は式に変数が2つ以上ある場合に、すべての変数の微分をベクトルにまとめること
# この関数はxの配列が2つの場合にのみ使用できる。
def numerical_gradient(func,x):
    print("numerical_gradient() x=",x)
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # func(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = func(x)

        # func(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = func(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # ガウス分布で初期化

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y = activation_func_library.softmax(z)
        loss = loss_func_library.cross_entropy_error(y,t)
        return loss

net = simpleNet()
print(net.W) # 重み
x = np.array([0.6,0.9]) # 入力データ
p = net.predict(x)
print(p) # 予測結果
np.argmax(p) # 最大値のインデックス

t = np.array([0,0,1]) # 正解ラベル

def f(W):
    print("f() W=",W)
    print("f() x=",x)
    return net.loss(x, t)

dW = numerical_gradient(f,net.W)
