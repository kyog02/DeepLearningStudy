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
                 weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size) # バイアスは最初は0
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size) # バイアスは最初は0
        
        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = activation_func_layers.AffineLayer(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = activation_func_layers.ReluLayer()
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
        return self.lastLayer.forward(y,t)

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