import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0,0.5])
w = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
a = np.dot(x,w)
b = np.array([0.1,0.2,0.3])
a2 = np.dot(x,w)+b
print("np.dot(x,w) = ",a)
print("np.dot(x,w) + b = ",a2)