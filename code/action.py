import torch
import torch.nn as nn
import numpy as np

from data import TrainSet, Action
from network import RNN
from torch.utils.data import DataLoader
from plot import Plot

n = 30
LR = 0.0001
EPOCH = 0
train_end = -500
# 数据集建立

r'''
df：训练使用
df_all：所有数据
df_index：标签
'''

df, df_all, df_index = Action.read_data('high', n=n, train_end=train_end)

df_all = np.array(df_all.tolist())
fig1 = Plot(1)
fig1.plot(df_index, df_all, label='real-data')

df_numpy = np.array(df)

# 归一
df_numpy_mean = np.mean(df_numpy)
df_numpy_std = np.std(df_numpy)

df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
df_tensor = torch.Tensor(df_numpy)

trainset = TrainSet(df_tensor)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

rnn = torch.load('rnn.pkl')

# rnn = RNN(n)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

for step in range(EPOCH):
    for tx, ty in trainloader:
        output = rnn(torch.unsqueeze(tx, dim=0))
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
    print(step, loss)
    if step % 10:
        torch.save(rnn, 'rnn.pkl')
torch.save(rnn, 'rnn.pkl')

generate_data_train = []
generate_data_test = []

test_index = len(df_all) + train_end

df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
df_all_normal_tensor = torch.Tensor(df_all_normal)
for i in range(n, len(df_all)):
    x = df_all_normal_tensor[i - n:i]
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = rnn(x)
    if i < test_index:
        generate_data_train.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
    else:
        generate_data_test.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
fig1.plot(df_index[n:train_end], generate_data_train, label='generate_train')
fig1.plot(df_index[train_end:], generate_data_test, label='generate_test')
fig1.save()
fig1.show()
fig2 = Plot(2)
fig2.plot(df_index[train_end:-400], df_all[train_end:-400], label='real-data')
fig2.plot(df_index[train_end:-400], generate_data_test[:-400], label='generate_test')
fig2.title('上证指数（指数代码：000001）')
fig2.save()
fig2.show()
