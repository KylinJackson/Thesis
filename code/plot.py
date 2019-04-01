import matplotlib.pyplot as plt


class Plot:
    def __init__(self, index):
        self.index = index

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
