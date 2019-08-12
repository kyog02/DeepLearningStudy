import numpy as np
import matplotlib.pylab as plt
# 損失関数ライブラリ

# 2乗和誤差(SSE)
# ソフトマックス関数の出力に対する学習に使える
# 回帰問題に使うのが一般的
def sum_of_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)


# 交差エントロピー誤差（CEE)
# np.log(0)が発生してしまうと、マイナスの無限大の値になってしまうためにそれを防ぐためのデルタ加算
# 分類問題に使うのが一般的
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum( t + np.log( y + delta ))