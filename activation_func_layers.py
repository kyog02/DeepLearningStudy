# 誤差逆伝播法を使ったニューラルネットワークで使用する
# 活性化関数レイヤ
# 活性化関数の逆伝播(backward)は、重み、バイアスの勾配を直接求めない。
# 直接、重み・バイアスの勾配を求めるのは順伝播で重み・バイアスを直接使用するAffineレイヤ
import numpy as np

from util import im2col
from util import col2im

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

# ドロップアウトレイヤ
# 過学習抑制する手法の一つ
# ニューロンをランダムに消去しながら学習する手法
# 訓練時に、隠れ層のニューロンをランダムに選び出し、その選び出したニューロンを消去する。
# テスト時には、すべてのニューロンの信号を伝達するが、各ニューロンの出力に対して、訓練時に消去した割合を乗算して出力する。
# 消去する判断のレシオはハイパーパラメータである。
# 活性化関数レイヤの後に入れる。(最後の出力層の後には不要)
# メリット：学習時にニューロンをランダムに消去することで、毎回異なるモデルを学習させていると解釈できる。アンサンブル学習と似た手法
# 補足：バッチ正規化と一緒に使うことは少ないらしい。
class DropoutLayer:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.15):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # 訓練時は、xと同じ形状の配列をランダムに生成して、レシオよりも小さかった要素を消去する。
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # テスト時は、訓練時に消去した割合を乗算して出力
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


# 畳み込み層(Convolution)
class ConvolutionLayer:
    def __init__(self,W,b,stride=1,pad=0):
        self.W = W  # フィルター(重み) (FN,C,FH,FW)の4次元配列
                    # FN:フィルターの個数
                    # C：チャンネル数
                    # FH：フィルタの縦幅
                    # FW：フィルタの横幅
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self,x):
        FN,C,FH,FW = self.W.shape   # フィルターのパラメータ。次元順に分割して格納
        N,C,H,W = x.shape           # 入力データのパラメータ。次元順に分割して格納
        out_h = int((H + 2*self.pad - FH) / self.stride + 1)
        out_W = int((W + 2*self.pad - FW) / self.stride + 1)

        col   = im2col(x,FH,FW,self.stride,self.pad)
        col_w = self.W.reshape(FN,-1).T # フィルターの展開
        out = np.dot(col,col_w) + self.b

        out = out.reshape(N,out_h,out_W,-1).transpose(0,3,1,2)

        # 逆伝播に使用
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class PoolingLayer:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展開
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        # Maxプーリング
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

# バッチ正規化レイヤ(BatchNormalization)
# 重みの初期値を適切に設定しないと、各層のアクティベーション分布が広がらないという問題に対して
# 各層で適度な広がりを持つように、強制的にアクティベーションの分布を調整しようとする処理である。
# メリット：
#  １．学習を早く進行させることができる。
#　２．初期値にそれほど依存しない
#　３．過学習を抑制する。(Dropoutなどの必要性を減らす)
# 隠れ層のAffineレイヤと活性化関数レイヤの間に入れる。(最後の出力層前は不要)
# 学習を行う際のミニバッチを単位として、ミニバッチごとに正規化を行う。
class BatchNormalizationLayer:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    # 訓練データかどうかで処理が変わる。
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
    
    # TODO:計算式が全然わからず。。
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            # 6章(CNN前)ではこちらの処理を通るはず。
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
