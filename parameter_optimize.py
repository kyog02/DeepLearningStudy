# ハイパーパラメータの更新にかかわるライブラリ

import numpy as np

# 確率的勾配降下法(SGD)
# パラメータの勾配(微分)を使って、勾配方向にパラメータを更新する。
# 関数の形状が等方的でない関数には、非効率のため使用しない。
# ⇒勾配の方向が、本来の最小値ではない方向を指していることが根本原因
# η(イータ)：学習係数
# W ← W - η * (∂L / ∂W)
class SGD:
    def __init__(self,lr=0.01):
        self.lr = lr

        def update(self,params,grads):
            for key in params.keys():
                params[key] -= self.lr * grads[key]

# モーメンタム
# モーメンタムとは運動量という意味
# SGDの更新経路はジグザグ動くことがあるが、モーメンタムは
# ボールが地面の傾斜を転がるように更新する。
# ⇒x軸方向に受ける力はとても小さいが、常に同じ方向の力を受けるため、同じ方向へ一定して加速することになるから
#   y軸方向には受ける力は大きいが、正と負の方向の力を交互に受けるため、y軸方向の速度は安定しない。
#   それにより、SGDに比べて、x軸方向へ早く近づくことができ、ジグザグの動きを軽減できる。
# v：速度
# η(イータ)：学習係数
# α(momentum)：運動量
# v ← αv - η * (∂L / ∂W)
# W ← W + v
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * self.grads[key]
            params[key] += self.v[key]

# AdaGrad
# 学習係数の減衰を実現
# 学習係数が小さいと学習に時間がかかりすぎる。逆に大きいと、発散して正しい学習が行えない。
# AdaGradはパラメータの要素ごとに適応的に学習係数を調整しながら学習を行う。
# h ← h + ∂L / ∂W Θ ∂L / ∂W
# W ← W - η * (1/√h) * (∂L / ∂W)
# 欠点：過去の勾配を2乗和としてすべて記録するため、学習を進めれば進めるほど、更新度合いは小さくなる。
#      ⇒改善手法としてRMSPropがある。
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            # 0除算対策で1e-7という小さい値を加算している
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

#=================================================================
# 以下、参照


# RMSprop
# 過去全ての勾配を均一に加算せずに、過去の勾配を徐々に忘れて、
# 新しい勾配の情報が大きく反映されるように加算します。
# 指数移動平均という
class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# Adam
# ボールが転がるように、物理法則に準じたMomentamと
# パラメータの要素ごとに更新ステップを調整したAdaGradの融合
# η：学習係数
# beta1：一次モーメント用係数(初期値0.9)
# beta2：二次モーメント用係数(初期値0.999)
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
