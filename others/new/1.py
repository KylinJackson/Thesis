import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import numpy as np
from data import DataSet
from network import LSTM, DA
from evaluate import Evaluate
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def generate_df(filename, affect, standardize=True):
    # 从csv文件读取数据
    df = pd.read_csv('data/{}'.format(filename), index_col='TrdDt')

    # 把时间作为index
    df.index = list(
        map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'),
            df.index)
    )

    # 取出要操作的列
    df_all = df['ClPr'].copy()

    df_input = pd.DataFrame()
    if standardize:
        # 归一
        mean = df_all.mean()
        std = df_all.std()
        df_standardized = (df_all - mean) / std

        # 重新调整df结构
        for i in range(affect):
            df_input[i] = list(df_standardized)[i:-(affect - i)]
        df_input['y'] = list(df_standardized)[affect:]
        df_input.index = df.index[affect:]
        df_index = list(df_input.index)

        return {
            'dataset': DataSet(torch.tensor(np.array(df_input)).cuda()),  # 返回Dataset形式的数据
            'real_data': df_all,  # 返回所有数据
            'mean': mean,  # 返回数据的平均值
            'std': std,  # 返回数据的
            'index': df_index,
        }

    else:
        for i in range(affect):
            df_input[i] = list(df_all)[i:-(affect - i)]
        df_input['y'] = list(df_all)[affect:]
        df_input.index = df.index[affect:]
        df_index = list(df_input.index)

        return {
            'dataset': DataSet(torch.tensor(np.array(df_input)).cuda()),  # 返回Dataset形式的数据
            'real_data': df_all,  # 返回所有数据
            'index': df_index,
        }


# hyper parameters
AFFECT = 5
EPOCH = 10000
STOCK = '上证指数'
PERIOD = '2009-2014'
BATCH = 20
TRAIN_FILE = STOCK + PERIOD + '_train.csv'
TEST_FILE = STOCK + PERIOD + '_test.csv'
STANDARDIZE = True
# train

data_train = generate_df(TRAIN_FILE, AFFECT, STANDARDIZE)
data_test = generate_df(TEST_FILE, AFFECT, STANDARDIZE)

data_loader_train = DataLoader(data_train['dataset'], batch_size=BATCH, shuffle=False)
data_loader_test = DataLoader(data_test['dataset'], batch_size=1, shuffle=False)
net = LSTM(AFFECT).cuda()
optimizer = optim.Adam(net.parameters())
loss_func = nn.MSELoss().cuda()
# loss_func = DA().cuda()

da_list = list()
mse_list = list()
theil_list = list()
l1_list = list()
mape_list = list()
r_list = list()

for step in range(EPOCH):
    loss = None
    for tx, ty in data_loader_train:
        output = net(tx.reshape(1, BATCH, AFFECT))
        loss = loss_func(output.reshape(BATCH), ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predict = list()

    for tx, ty in data_loader_test:
        output = net(tx.reshape(1, 1, AFFECT))
        output = output.reshape(1).detach()
        if STANDARDIZE:
            predict.append(float(output) * data_test['std'] + data_test['mean'])
        else:
            predict.append(float(output))
    if (step + 1) % 100 == 0:
        evaluator = Evaluate(data_test['real_data'][AFFECT:], predict)
        da_list.append(evaluator.DA())
        mse_list.append(evaluator.MSELoss())
        theil_list.append(evaluator.Theil_U())
        l1_list.append(evaluator.L1Loss())
        mape_list.append(evaluator.MAPE())
        r_list.append(evaluator.R())

    print('step: {}, data accuracy: {}'.format(step + 1, 1 - loss.detach()))

    if (step + 1) % 1000 == 0:
        plt.cla()
        zh_font = fm.FontProperties(fname='font/simhei.ttf')
        plt.plot(data_test['index'], data_test['real_data'][AFFECT:], label='real data')
        plt.plot(data_test['index'], predict, label='predict data')
        plt.title('{}.png'.format(step + 1), fontproperties=zh_font)
        plt.savefig('fig/{}.png'.format(step + 1))
        plt.cla()

np.save('da.npy', np.array(da_list))
np.save('mse.npy', np.array(mse_list))
np.save('theil.npy', np.array(theil_list))
np.save('l1.npy'.format(), np.array(l1_list))
np.save('mape.npy'.format(), np.array(mape_list))
np.save('r.npy'.format(), np.array(r_list))