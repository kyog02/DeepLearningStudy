import numpy as np
import matplotlib.pylab as plt

def ReLU(x):
    return np.maximum(0,x)

x = np.arange(-5.0,5.0,0.1)
y = ReLU(x)
plt.plot(x,y)
plt.ylim(-0.1,5.1)
plt.show()
