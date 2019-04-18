import time

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, index, t):
        self.index = index
        self.zh_font = fm.FontProperties(fname='font/simhei.ttf')
        self.t = time.strftime('%Y-%m-%d %H:%M:%S', t)

    def plot(self, data_index, data, label=None):
        plt.figure(self.index)
        if label is None:
            plt.plot(data_index, data)
        else:
            plt.plot(data_index, data, label=label)
        plt.legend()

    def save(self, filename):
        plt.figure(self.index)
        plt.savefig('fig/{}-{}.png'
                    .format(filename,
                            self.t))

    def title(self, title, zh=False):
        plt.figure(self.index)
        if zh:
            plt.title(title, fontproperties=self.zh_font)
        else:
            plt.title(title)

    @staticmethod
    def show():
        plt.show()
