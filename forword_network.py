# ライブラリ
import numpy as np

# 自作ファイル
import activation_func_library

# forward_network.py
# ニューラルネットワークのフォワード方向
# こちらでは、学習は完了した前提として、重み、バイアスは自分で決めている。


# 重み、バイアスの初期化(設定)
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network,x):
    # 重み、バイアスの取得
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    # 計算式
    a1 = np.dot(x,W1) + b1
    z1 = activation_func_library.sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = activation_func_library.sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = activation_func_library.identity_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)