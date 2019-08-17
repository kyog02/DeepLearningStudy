import numpy as np
import matplotlib.pylab as plt
# 活性化関数ライブラリ


# ステップ関数
# パーセプトロンで使用
# 信号が急に変更する活性化関数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# シグモイド関数
# 計算式：y = 1/(1+exp(-x))
# 入力信号を曲線にして出力する活性化関数(信号がなめらかに変更する関数)
# ニューラルネットワークで使用
# 2クラス分類問題で使用するのが一般的
# 0～1の間にマッピングしなおして、出力する。
def sigmoid(x):
    return 1 / (1+np.exp(-x))

# シグモイド関数の微分
# 入力(y)：シグモイド関数の出力結果
# 計算式：dx = (1 - y) * y
def sigmoid_differential(y):
    return (1 - y) * y

# tanh関数
# 計算式：(exp(x) - exp(-x))/(exp(x) + exp(-x))
# -1～1の間にマッピングして出力する。
# 出力の中が0である
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# tanh関数の微分
# 計算式：4 / ((exp(x) + exp(-x)) ** 2)
# 入力：tanh関数の入力値
# TODO:xが↑であっているか、それとも前レイヤの微分出力値(dout)か要確認
def tanh_differential(x):
    return 4 / ((np.exp(x) + np.exp(-x)) ** 2)

# 恒等関数
# 入力信号をそのまま出力する活性化関数
# 回帰問題の出力層の活性化関数として使用するのが一般的
def identity_function(x):
    return x

# ソフトマックス関数
# y = exp(x)/sum(exp(x))
# 補足：実際には、指数計算(exp)を行うと、すぐに表現できない値まで大きくなってしまう(オーバーフロー)
#      そのため、入力信号の各ニューロンに何らかの定数を引き算(もしくは足し算)して、値を小さくする。
#      exp(x+定数C)/sum(exp(x+定数C))
#      定数は入力信号の中で一番大きい値にするのが一般的
# 出力信号の各ニューロンに対して、全ての入力信号が影響を与える
# 出力値は0〜1.0の間の実数になり、出力の総和は1になる。
# ↑の性質のおかげで、ソフトマックスの出力を確率として解釈することができる。
# 注意点：ソフトマックスを適用しても、各要素の大小関係に影響はない。指数関数が単調増加する関数であるため。
# 分類問題の出力層の活性化関数として使用するのが一般的
# ★要素の大小関係が変わらないなら、出力層にソフトマックス適用しようがしまいが、判定結果に変わりはない。
#  出力層にソフトマックスが必要な理由はニューラルネットワークの学習に関係がある(TODO)
def softmax(x):
    # c = np.max(x)
    # exp_x = np.exp(x - c)
    # sum_exp_x = np.sum(exp_x)
    # y = exp_x / sum_exp_x
    # return y
    # なぜ転置が必要？？⇒オーバーフロー対策で、入力信号の中から最大値を引く必要があるため
    # 転置せず、np.max(axis=1)にすることでも、入力信号の最大値の取得はできるが、そのままだとxとの引き算ができなくなるため、扱いやすいよう転置している。
    if x.ndim == 2: # 画像は2次元
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

# ReLU関数
# シグモイド関数に比べて早く収束する
# ユニットが活発化している領域では、常に微分が大きい
# 勾配は大きいだけでなく一定
# 活性化値が0になるような事例が勾配に基づく手法では学習できない
# ｘ＜＝０の時は関数の値も勾配も0になる。そのため、一度不活性になると、学習の間はずっと不活性
def ReLU(x):
    return np.maximum(0,x)

# ReLU関数の微分
# 注意：xは順伝播時の入力値
def ReLU_differential(x):
    return np.where( x > 0,1,0)

# Leaky ReLU関数
# xが正の値の場合はx
# xが負の値の場合は0.01x(負の場合は傾きを緩くする)
# 飽和せず、計算効率がよい
# シグモイドやtanhに比べて早く収束する
# xが０以下の時も、わずかな傾きを持つ
def LeakyReLU(x):
    return np.maximum(0.01*x,x)

# Leaky ReLU関数の微分
def LeakyReLU_differential(x):
    return np.where(x > 0,1,0.01)