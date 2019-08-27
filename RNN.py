import numpy as np


class RNN:
    def __init__(self,Wx,Wh,b):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        # 逆伝播時の計算に使用する中間データをキャッシュとして用意する。
        self.cache = None


    # RNNの順伝播は ht = tanh((ht-1 * Wh) + (xt * Wx) + b)
    # レイヤとしては、行列積(Matmul)と加算⇒活性化関数tanhでできている。
    # 式の項目：変数名 ⇒ ht-1:h_prev, Wh:Wh, xt:x , Wx:Wx, b:b
    # h_prevは最初の値どうするんやろ。。順伝播の出力結果
    def forward(self,x,h_prev):
        Wx,Wh,b = self.params
        t = np.dot(h_prev,Wh) * np.dot(x,Wx) + b
        h_next = np.tanh(t)
        self.cache = (x,h_prev,h_next)
        return h_next #ht出力

    # 逆伝播
    # dh_nextは何の値？TimeRNNの出力結果か。。？またSoftmaxWithLossみたいなのがあるのか。。？
    def backward(self,dh_next):
        Wx,Wh,b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)    # tanhの逆伝播：dout * (1 - y**2)
        db = np.sum(dt, axis=0)             # Affineの逆伝播(バイアス)

        # RNNの順伝播：ht = tanh((ht-1 * Wh) + (xt * Wx) + b)の(ht-1 * Wh)側の逆伝播
        # h_prevが入力データとして考える。なので、h_prevをxに置き換えたらいつもの逆伝播と一緒
        dWh = np.dot(h_prev.T, dt)          # dW = np.dot(x.T,dout)の考え方。重みの勾配は入力値から求める。(逆になる)
        dh_prev = np.dot(dt, Wh.T)          # dx = np.dot(dout,W.T)の考え方。入力値の勾配は重みから求める。(逆になる)

        # RNNの順伝播：ht = tanh((ht-1 * Wh) + (xt * Wx) + b)の(xt * Wx)側の逆伝播
        dWx = np.dot(x.T, dt)               # dW = np.dot(x.T,dout)の考え方
        dx = np.dot(dt, Wx.T)               # dx = np.dot(dout,W.T)の考え方

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx,dh_prev # 入力データが2つ
    
class TimeRNN:
    def __init__(self,Wx,Wh,b,stateful=False):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers = None

        self.h  = None  # forward()の最後のRNNレイヤの隠れ状態を保持する。
        self.dh = None  # backward()を呼んだ時に一つ前のブロックへの隠れ状態の勾配を保持します。
        self.stateful = stateful # 隠れ状態を維持するかどうかのboolean。falseの場合、TimeRNNのforward()が呼ばれるたびに消去する。

    # 隠れ状態を設定するメソッド。拡張性考慮
    def set_state(self,h):
        self.h = h
    # 隠れ状態をリセットするメソッド。拡張性考慮
    def reset_state(self):
        self.h = None

    def forward(self,xs):
        Wx,Wh,b = self.params
        N,T,D = xs.shape
        D,H = Wx.shape # D:入力ベクトルの次元数

        self.layers = []
        hs = np.empty((N,T,H),dtype='f')

        if not self.stateful or self.h is None
            self.h = np.zeros((N,H),dtype='h')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:],self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs

    def backward(self,dhs):
        Wx,Wh,b = self.params
        N,T,H = dhs.shape
        D,H = Wx.shape

        dxs = np.empty((N,T,D),dtype='f')
        dh = 0
        grads = [0,0,0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx,dh = layer.backward(dhs[:,t,:] + dh) # 合算した勾配
            dxs[:,t,:] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs