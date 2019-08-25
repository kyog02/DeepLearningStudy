# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict

# 自作ファイル
from gradient import numerical_gradient_impl
import activation_func_layers

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,
                 weight_init_std=0.01,weight_decay_lambda=0.1):

        # 引数受け取り
        self.weight_decay_lambda = weight_decay_lambda

        # 重みの初期化(標準偏差使用)
        # 重みの初期化を適切な設定方法で行わないと、活性化関数の出力結果が偏ってしまい、
        # 学習(逆伝播)の際に、勾配消失が問題になってしまう。
        # 各層でデータの広がりが均一に分布されていると、逆伝播の際も適切な値が流れることが期待できる。
        self.params = {}
        #self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        #self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b1'] = np.zeros(hidden_size) # バイアスは最初は0
        self.params['b2'] = np.zeros(output_size) # バイアスは最初は0
        
        # sigmoid関数/tanh関数などS字カーブの活性化関数使用時にはXavierの初期値を使用するのが一般的
        #self.params['W1'] = np.random.randn(input_size,hidden_size) / np.sqrt(input_size)   # 平方根に渡すのはインプット側の層
        #self.params['W2'] = np.random.randn(hidden_size,output_size) / np.sqrt(hidden_size) # 平方根に渡すのはインプット側の層

        # ReLU関数の活性化関数使用時にはHeの初期値を使用するのが一般的
        self.params['W1'] = np.random.randn(input_size,hidden_size) / np.sqrt(input_size) * np.sqrt(2)      # 平方根に渡すのはインプット側の層
        self.params['W2'] = np.random.randn(hidden_size,output_size) / np.sqrt(hidden_size) * np.sqrt(2)    # 平方根に渡すのはインプット側の層

        # バッチ正規化のベータとガンマの値初期化
        self.params['gamma'] = np.ones(hidden_size) # 隠れ層のサイズ分初期化
        self.params['beta'] = np.zeros(hidden_size) # 隠れ層のサイズ分初期化

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = activation_func_layers.AffineLayer(self.params['W1'],self.params['b1'])
        self.layers['BatchNorm'] = activation_func_layers.BatchNormalizationLayer(self.params['gamma'], self.params['beta']) 
        self.layers['Relu1'] = activation_func_layers.ReluLayer()
        # ドロップアウトはお試しでは入れても、意味がなかった。
        #self.layers['Dropout'] = activation_func_layers.DropoutLayer()
        self.layers['Affine2'] = activation_func_layers.AffineLayer(self.params['W2'],self.params['b2'])
        self.lastLayer = activation_func_layers.SoftmaxWithLossLayer()

    # 推論処理
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 損失関数
    def loss(self,x,t):
        y = self.predict(x)
        # 重み減衰取得
        weight_decay = self.__weight_decay()
        return self.lastLayer.forward(y,t) + weight_decay

    # L2正則化。重み減衰
    # 過学習抑制のために使用する。そもそも過学習の原因は重みが大きい値をとることにより発生することが多々ある。
    # ラムダはハイパーパラメータの一つだが、こちらを大きく設定すればするほど、大きな重みを取ることに対して強い減衰を加えることになる。
    # (ラムダは正則化の効果の重み)
    # 計算式：損失関数の出力結果 + 1/2 * λ(ラムダ) * Wの二乗
    #        (すべての重みに対して実施する)
    def __weight_decay(self):
        weight_decay = 0
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params['W1'] ** 2)
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params['W2'] ** 2)
        return weight_decay


    # 認識制度
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t,axis=1) #TODO

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    # 数値微分の勾配により勾配を求める
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient_impl(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_impl(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_impl(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_impl(loss_W, self.params['b2'])
        return grads

    # 誤差逆伝播法により、勾配を求める
    def backprop_gradient(self,x,t):
        # forward
        self.loss(x,t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads