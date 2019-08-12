# 恒等関数
# 入力信号をそのまま出力する活性化関数
# 回帰問題で使用するのが一般的
import numpy as np
import matplotlib.pylab as plt

def identity_function(x):
    return x

x = np.arange(-5.0,5.0,0.1)
y = identity_function(x)
plt.plot(x,y)
plt.ylim(-6.1,6.1)
plt.show()
