import numpy as np

# f = 損失関数
# x = 重み or バイアス
def numerical_gradient_impl(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = float(tmp) + h
        fxh1 = f(x)

        x[idx] = float(tmp) - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp # 値を元に戻す
        it.iternext()

    return grad

