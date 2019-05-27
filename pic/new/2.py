import numpy as np
import matplotlib.pyplot as plt

x_index = np.arange(100, 10100, 100)
da_list = np.load('da.npy')
l1_list = np.load('l1.npy')
mape_list = np.load('mape.npy')
mse_list = np.load('mse.npy')
r_list = np.load('r.npy')
theil_list = np.load('theil.npy')

plt.cla()
plt.suptitle('Influence by Coefficient')
plt.subplot(331)
plt.plot(x_index, da_list)
plt.title('Data Accuracy')
plt.xlabel('epoch')
plt.ylabel('Data Accuracy')

plt.subplot(332)
plt.plot(x_index, mse_list)
plt.title('MSE Loss')
plt.xlabel('epoch')
plt.ylabel('MSE Loss')

plt.subplot(333)
plt.plot(x_index, theil_list)
plt.title('Theil U')
plt.xlabel('epoch')
plt.ylabel('Theil U')

plt.subplot(334)
plt.plot(x_index, l1_list)
plt.title('L1 Loss')
plt.xlabel('epoch')
plt.ylabel('L1 Loss')

plt.subplot(335)
plt.plot(x_index, mape_list)
plt.title('MAPE')
plt.xlabel('epoch')
plt.ylabel('MAPE')

plt.subplot(336)
plt.plot(x_index, r_list)
plt.title('R')
plt.xlabel('epoch')
plt.ylabel('R')

plt.subplots_adjust(wspace=1, hspace=3)
plt.savefig('fig/test.png')
plt.show()
plt.cla()
