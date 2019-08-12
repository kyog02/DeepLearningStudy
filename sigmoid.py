# シグモイド関数
# 入力信号を曲線にして出力する活性化関数
# 2クラス分類問題で使用するのが一般的
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
