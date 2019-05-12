import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (math.exp(-x) + 1)


def ReLU(x):
    if x > 0:
        return x
    else:
        return 0


X = np.arange(-10, 10, 0.01)
Y = list()
for i in X:
    Y.append(ReLU(i))

if __name__ == '__main__':
    plt.figure(1)
    plt.plot(X, Y, label='ReLU')
    plt.title('ReLU')
    plt.xlabel('z')
    plt.ylabel('f(z)')
    plt.savefig('ReLU.png')
    plt.show()
    plt.cla()
