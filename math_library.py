import numpy as np
import matplotlib.pylab as plt


# 微分
# 誤差を含んでいないため、使用しない
def diff(func,x):
    h = 1e-4
    return (func(x+h) - func(x))/h


# 数値微分
# 数値微分には誤差が含まれるため、
def numrerical_diff(func,x):
    h = 1e-4 # 0.0001
    return (func(x+h) - func(x-h)) / (2*h)


# 勾配は式に変数が2つ以上ある場合に、すべての変数の微分をベクトルにまとめること
# この関数はxの配列が2つの場合にのみ使用できる。
def numerical_gradient(func,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # func(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = func(x)

        # func(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = func(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad



# sample(簡単な二次間数)
# y = 0.01x^2 + 0.1x
def func1(x):
    return 0.01 * x **2 + 0.1 * x


# x = np.arange(0.0,20.0,0.1)
# y = func1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x,y)
# plt.show()

def func1_sample(t):
    return t**2

print(diff(func1_sample,0))
# print(numrerical_diff(func1,10))

# sample2(引数の二乗和を計算する)
# 変数が二つある
# f(x0,x1) = x0^2 + x1^2
def func2(x):
    return x[0]**2 + x[1]**2
    # またはnp.sum(x**2)

def func2_tmp1(x0):
    return x0**2.0 + 4.0**2.0

def func2_tmp2(x1):
    return 3.0**2.0 + x1**2.0


# 偏微分は式に変数が2つ以上ある場合に、1つの変数の微分を求めること
# print(numrerical_diff(func2_tmp1,3.0))
# print(numrerical_diff(func2_tmp2,4.0))

# print(numerical_gradient(func2,np.array([3.0,4.0,3.0])))
# print(numerical_gradient(func2,np.array([0.0,2.0])))