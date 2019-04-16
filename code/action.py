import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import *
from plot import Plot
from network import LSTM

network_name = 'LSTM'
TRAIN = False
SHOW = False
AFFECT = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LR = 0.0001
EPOCH = 1000
TRAIN_END = -500
FILENAME = 'data_train.csv'
COLUMN = 'HiPr'
INDEX_COL = 'TrdDt'
BATCH_SIZE = 10

r'''
df：训练使用
df_all：所有数据
df_index：标签
'''
df, df_all, df_index = Action.generate_df(
    FILENAME,
    COLUMN,
    INDEX_COL,
    AFFECT,
    TRAIN_END
)

df_numpy = np.array(df)

# 归一
df_numpy_mean = np.mean(df_numpy)
df_numpy_std = np.std(df_numpy)

df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
df_tensor = torch.Tensor(df_numpy)

train_set = TrainSet(df_tensor)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

if TRAIN:
    lstm = LSTM(AFFECT, HIDDEN_SIZE, NUM_LAYERS)
    optimizer = optim.Adam(lstm.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for step in range(EPOCH):
        loss = None
        for tx, ty in train_loader:
            output = lstm(tx.reshape(1, BATCH_SIZE, AFFECT))
            loss = loss_func(output.reshape(BATCH_SIZE), ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('step: {}, loss: {}'.format(step + 1, loss.detach()))
    torch.save(lstm, 'save/{}.pt'.format(network_name))
else:
    lstm = torch.load('save/{}.pt'.format(network_name))

generate_data_train = list()
generate_data_test = list()

test_index = len(df_all) + TRAIN_END

df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
df_all_normal_tensor = torch.Tensor(df_all_normal)
for i in range(AFFECT, len(df_all)):
    x = df_all_normal_tensor[i - AFFECT:i]
    x = x.reshape(1, 1, AFFECT)
    y = lstm(x)
    y = y.reshape(1).detach()
    if i < test_index:
        generate_data_train.append(y.numpy() * df_numpy_std + df_numpy_mean)
    else:
        generate_data_test.append(y.numpy() * df_numpy_std + df_numpy_mean)

plt1 = Plot(1)
plt1.plot(df_index, df_all, 'real_data')
plt1.plot(df_index[AFFECT:TRAIN_END], generate_data_train, 'generate_train')
plt1.plot(df_index[TRAIN_END:], generate_data_test, 'generate_test')
plt1.save('fig1')

plt2 = Plot(2)
plt2.title('上证指数', zh=True)
plt2.plot(df_index[TRAIN_END:-400], df_all[TRAIN_END:-400], 'real-data')
plt2.plot(df_index[TRAIN_END:-400], generate_data_test[:-400], 'generate_test')
plt2.save('fig2')

plt3 = Plot(3)
plt3.plot(df_index[TRAIN_END:], df_all[TRAIN_END:], 'real-data')
plt3.plot(df_index[TRAIN_END:], generate_data_test, 'generate_test')
plt3.save('fig3')

if SHOW:
    Plot.show()
