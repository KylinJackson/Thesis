import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, index):
        self.index = index
        self.zh_font = fm.FontProperties(fname='font/simhei.ttf')

    def plot(self, data_index, data, label):
        plt.figure(self.index)
        plt.plot(data_index, data, label=label)
        plt.legend()

    def show(self):
        plt.figure(self.index)
        plt.show()

    def save(self, filename):
        plt.figure(self.index)
        plt.savefig('fig/{}.png'.format(filename))

    def title(self, title, zh=False):
        plt.figure(self.index)
        if zh:
            plt.title(title, fontproperties=self.zh_font)
        else:
            plt.title(title)
