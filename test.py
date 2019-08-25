import numpy as np
import matplotlib.pyplot as plt

# x = np.array([1.0,0.5])
# w = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
# a = np.dot(x,w)
# b = np.array([0.1,0.2,0.3])
# a2 = np.dot(x,w)+b
# print("np.dot(x,w) = ",a)
# print("np.dot(x,w) + b = ",a2)


# x = np.array([5.0,6.0])
# y = np.array([[2.0,4.0,6.0],[3.0,5.0,7.0]])
# b = np.random.rand(3)

# print(x)
# print(y)
# print(b)
# print(np.dot(x,y))

# a = np.array([[5,1],[1,2],[3,4]])
# print(a)
# b = np.array([[1,2,3,4],[5,6,7,8]])
# print(b)
# print(np.dot(a,b))

# A = np.array([[1,2],[3,4],[5,6]])
# B = np.array([[2],[3]])
# C = np.dot(A,B)
# print(A)
# print(B)
# print(C)

# np.random.randn()のテスト
#print(np.random.randn(3)) # 3行のデータを生成
#print(np.random.randn(5,3))# 5列3行のデータを生成

# 次元数の確認
# print("次元数の確認：=============================S")
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.arange(6).reshape((3, 2))
# c = np.array([b, b])
# print("a:" + str(a))
# print("a.ndim:" + str(a.ndim))
# print("b:" + str(b))
# print("b.ndim:" + str(b.ndim))
# print("c:" + str(c))
# print("c.ndim:" + str(c.ndim))
# print("次元数の確認：=============================E")


#CNN

x = np.random.rand(10,1,28,28) # フィルタ数10個1チャンネル高さ28幅28の4次元配列
y = x[0,0]

a = [2,3]
b = np.pad(a,[1,2],"constant")
print(b)