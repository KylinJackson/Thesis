import time
import plot

import test
import train

TRAIN = True
TEST = True

NETWORK = 'LSTM'
# AFFECT = 10
STOCK_NAME = input('stock name: ')
TIME = input('period: ')
TITLE = STOCK_NAME
STOCK_NAME += TIME
TEST_FILENAME = STOCK_NAME + '_test.csv'
TRAIN_FILENAME = STOCK_NAME + '_train.csv'
TRAIN_BATCH = 5
COLUMN = 'ClPr'
TIME_NOW = time.localtime()
LR = 0.0001
EPOCH = 2000

result = list()
for i in range(5, 40, 5):
    AFFECT = i
    if TRAIN:
        train.train(TRAIN_FILENAME,
                    affect=AFFECT,
                    network=NETWORK,
                    title=TITLE,
                    train_batch=TRAIN_BATCH,
                    column=COLUMN,
                    lr=LR,
                    epoch=EPOCH)
    if TEST:
        evaluator = test.test(TEST_FILENAME,
                              TIME_NOW,
                              TITLE,
                              network=NETWORK,
                              affect=AFFECT,
                              column=COLUMN)
        result.append(evaluator)

da_list = []
mse_list = []
theil_list = []
l1_list = []
r_list = []
mape_list = []
index_list = []
for i in result:
    index_list.append(i + 1)
    da_list.append(float(i.DA()))
    mse_list.append(float(i.MSELoss()))
    theil_list.append(float(i.Theil_U()))
    l1_list.append(float(i.L1Loss()))
    r_list.append(float(i.R()))
    mape_list.append(float(i.MAPE()))

now = time.localtime()
da_plt = plot.Plot(1, now, 'LSTM')
da_plt.plot(index_list, da_list, 'DA')
da_plt.title('DA', zh=False)
da_plt.save('DA')
plot.Plot.cla()

mse_plt = plot.Plot(2, now, 'LSTM')
mse_plt.plot(index_list, mse_list, 'MSE')
mse_plt.title('MSE', zh=False)
mse_plt.save('MSE')
plot.Plot.cla()

theil_plt = plot.Plot(3, now, 'LSTM')
theil_plt.plot(index_list, theil_list, 'Theil U')
theil_plt.title('Theil U', zh=False)
theil_plt.save('Theil U')
plot.Plot.cla()

l1_plt = plot.Plot(4, now, 'LSTM')
l1_plt.plot(index_list, l1_list, 'L1 Loss')
l1_plt.title('L1 Loss', zh=False)
l1_plt.save('L1 Loss')
plot.Plot.cla()

r_plt = plot.Plot(5, now, 'LSTM')
r_plt.plot(index_list, r_list, 'R')
r_plt.title('R', zh=False)
r_plt.save('R')
plot.Plot.cla()

mape_plt = plot.Plot(6, now, 'LSTM')
mape_plt.plot(index_list, mape_list, 'MAPE')
mape_plt.title('MAPE', zh=False)
mape_plt.save('MAPE')
plot.Plot.cla()
