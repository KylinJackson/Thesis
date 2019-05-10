import os
import time

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, index, time_now, network):
        self.index = index
        self.zh_font = fm.FontProperties(fname='font/simhei.ttf')
        self.t = time.strftime('%Y-%m-%d-%H-%M-%S', time_now)
        self.network = network

    def plot(self, data_index, data, label=None):
        plt.figure(self.index)
        if label is None:
            plt.plot(data_index, data)
        else:
            plt.plot(data_index, data, label=label)
        plt.legend()

    def save(self, filename):
        plt.figure(self.index)
        if not os.path.exists('fig/{}'.format(self.network)):
            os.makedirs('fig/{}'.format(self.network))
        plt.savefig('fig/{}/{}-{}.png'.format(self.network,
                                              filename,
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

    @staticmethod
    def cla():
        plt.cla()
