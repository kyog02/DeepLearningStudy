# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict

# 自作ファイル
from gradient import numerical_gradient_impl
import activation_func_layers

class SimpleConvNet:
    def __init__(self,input_dim=(1,28,28),                              # 入力データのチャンネル、高さ、横幅の次元
                 conv_param={'filter_num':30,'filter_size':5,
                             'padding':0, 'stride':1},                  # フィルタ数、フィルタサイズ(正方形前提？)、ゼロパディング、ストライド
                 hidden_size=100,output_size=10, weight_init_std=0.01): # 隠れ層、出力層、重み係数
        filter_num  = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_padding  = conv_param['padding']
        filter_stride = conv_param['stride'] 
        input_size = input_dim[1] # 高さが入力サイズになる。正方形やから？
        conv_output_size = (input_size - filter_size + 2 * filter_padding) / filter_stride + 1 # 高さと横幅共通。正方形やから
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))# TODO:プーリング層の出力サイズの決め方はどうなってる？畳み込み層/2？ウインドウサイズとストライドは同じ値にしないといけない

        # 重みの初期化
        self.params = {}
        # W1,b1は畳み込み層で使用
        self.params['W1'] = weight_init_std * np.random.randn(filter_num,input_dim[0],filter_size,filter_size) # 重みW1：フィルタ数、チャンネル、フィルタのサイズ(高さ)、フィルタのサイズ(横幅)
        self.params['b1'] = np.zeros(filter_num) # 畳み込み層のバイアスはフィルタ数
        # W2,b2はAffineレイヤ1で使用(畳み込み層⇒ReLU⇒プーリング層の次)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,hidden_size) #入力サイズ(プーリング層の出力サイズ)と出力サイズ(活性化関数に渡すサイズ) 
        self.params['b2'] = np.zeros(hidden_size) # Affineレイヤのバイアスは出力側のサイズに合わせる
        # W3,b3はAffineレイヤ2で使用(畳み込み層⇒ReLU⇒プーリング層⇒Affine1⇒ReLUの次)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size,output_size) # 入力サイズ(活性化関数の入力サイズ)と出力サイズ
        self.params['b3'] = np.zeros(output_size) # Affineレイヤのバイアスは出力側のサイズに合わせる

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = activation_func_layers.ConvolutionLayer(self.params['W1'],
                                                                       self.params['b1'],
                                                                       self.params['stride'],
                                                                       self.params['padding'])
        self.layers['Relu1'] = activation_func_layers.ReluLayer()
        self.layers['Pool1'] = activation_func_layers.PoolingLayer(pool_h=2,pool_w=2,stride=2)
        self.layers['Affine1'] = activation_func_layers.AffineLayer(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = activation_func_layers.ReluLayer()
        self.layers['Affine2'] = activation_func_layers.AffineLayer(self.params['W3'], self.params['b3'])
        self.last_layer = activation_func_layers.SoftmaxWithLossLayer()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x):
        return self.last_layer.forward(y,t)

    def gradient(self,x,t):
        # forward
        self.loss(x,t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in self.layers.values():
            layer.backward(dout)

        # 勾配設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads