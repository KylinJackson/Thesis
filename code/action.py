import numpy as np
import torch
from torch.utils.data import DataLoader

from data import *
from plot import Plot

AFFECT = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LR = 0.0001
EPOCH = 0
TRAIN_END = -500
FILENAME = 'data_train.csv'
COLUMN = 'HiPr'
INDEX_COL = 'TrdDt'

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
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

lstm = torch.load('save/lstm.pt')

# lstm = LSTM(AFFECT, HIDDEN_SIZE, NUM_LAYERS)
# optimizer = optim.Adam(lstm.parameters(), lr=LR)
# loss_func = nn.MSELoss()
#
# for step in range(EPOCH):
#     loss = None
#     for tx, ty in train_loader:
#         output = lstm(torch.unsqueeze(tx, dim=0))
#         loss = loss_func(torch.squeeze(output), ty)
#         optimizer.zero_grad()  # clear gradients for this training step
#         loss.backward()  # back propagation, compute gradients
#         optimizer.step()
#     print(step, loss)
#     if step % 100:
#         torch.save(lstm, 'save/lstm.pt')
# torch.save(lstm, 'save/lstm.pt')

generate_data_train = []
generate_data_test = []

test_index = len(df_all) + TRAIN_END

df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
df_all_normal_tensor = torch.Tensor(df_all_normal)
for i in range(AFFECT, len(df_all)):
    x = df_all_normal_tensor[i - AFFECT:i]
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = lstm(x)
    if i < test_index:
        generate_data_train.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
    else:
        generate_data_test.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)

plt1 = Plot(1)
plt2 = Plot(2)
plt1.plot(df_index, df_all, 'real_data')
plt1.plot(df_index[AFFECT:TRAIN_END], generate_data_train, 'generate_train')
plt1.plot(df_index[TRAIN_END:], generate_data_test, 'generate_test')
# plt1.save('fig1')
# plt1.show()
plt2.title('上证指数', zh=True)
plt2.plot(df_index[TRAIN_END:-400], df_all[TRAIN_END:-400], 'real-data')
plt2.plot(df_index[TRAIN_END:-400], generate_data_test[:-400], 'generate_test')
# plt2.save('fig2')
# plt2.show()


f_out = open('save/da_test.txt', 'w')
a = np.array(df_all[TRAIN_END:-400])
b = np.array(generate_data_test[:-400])
ind = df_index[TRAIN_END:-400]
m = list()
for i in range(len(a)):
    m.append(a[i] - b[i])
    print(a[i] - b[i], file=f_out)
plt3 = Plot(3)
plt3.plot(ind, m)
plt3.title('差值', zh=True)
plt3.save('fig3')
plt3.show()

sum_abs = 0
for i in range(len(a)):
    sum_abs += abs(a[i] - b[i])
    print(i, sum_abs, file=f_out)
da = sum_abs / len(a)
print(da, file=f_out)
f_out.close()
