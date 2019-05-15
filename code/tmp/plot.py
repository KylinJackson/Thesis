import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return math.tanh(x)


def ReLU(x):
    if x > 0:
        return x
    else:
        return 0


x = np.arange(-5, 5, 0.01)
y_sigmoid = list()
y_tanh = list()
y_ReLU = list()
for i in x:
    y_sigmoid.append(sigmoid(i))
    y_tanh.append(tanh(i))
    y_ReLU.append(ReLU(i))

plt.figure(1)
plt.suptitle('Activation Function')
plt.subplot(221)
plt.title('Sigmoid Function')
plt.plot(x, y_sigmoid)
plt.xlabel('z')
plt.ylabel('f(z)')
plt.subplot(222)
plt.title('Tanh Function')
plt.plot(x, y_tanh)
plt.xlabel('z')
plt.ylabel('f(z)')
# plt.subplot(223)
# plt.title('ReLU Function')
# plt.plot(x, y_ReLU)
# plt.xlabel('z')
# plt.ylabel('f(z)')
plt.subplots_adjust(wspace=0.5)
plt.savefig('plot.png')
