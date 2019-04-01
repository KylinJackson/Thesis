import matplotlib.pyplot as plt
import matplotlib


class Plot:
    def __init__(self, index):
        self.index = index
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    def plot(self, data_index, data, label):
        plt.figure(self.index)
        plt.plot(data_index, data, label=label)
        plt.legend()

    def show(self):
        plt.figure(self.index)
        plt.show()

    def save(self):
        plt.figure(self.index)
        plt.savefig('fig{}.png'.format(self.index))

    def title(self, title):
        plt.figure(self.index)
        plt.title(title)
