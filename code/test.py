import time

import torch
from torch.utils.data import DataLoader

import globalvar as gl
from data import Action
from evaluate import Evaluate
from plot import Plot
from record import Logger

# 超参数设置
NETWORK_NAME = gl.get_value('network')
AFFECT = gl.get_value('affect')
FILENAME = gl.get_value('test_filename')
COLUMN = gl.get_value('column')
INDEX_COL = 'TrdDt'
BATCH_SIZE = 1
PLOT_NAME = ['fig1']

# 加载数据
data = Action.generate_df(
    FILENAME,
    COLUMN,
    INDEX_COL,
    AFFECT
)
data_loader = DataLoader(data['dataset'], batch_size=BATCH_SIZE, shuffle=False)

net = torch.load('save/{}.pt'.format(NETWORK_NAME))

predict = list()
for tx, ty in data_loader:
    output = net(tx.reshape(1, BATCH_SIZE, AFFECT))
    output = output.reshape(1).detach()
    predict.append(output * data['std'] + data['mean'])

plt1 = Plot(1)
plt1.plot(data['index'], data['real_data'][AFFECT:], 'real data')
plt1.plot(data['index'], predict, 'predict data')
plt1.save(PLOT_NAME[0])
Plot.show()

evaluator = Evaluate(data['real_data'][AFFECT:], predict)

logger = Logger('test.log')
basic_info = 'tested {}.'.format(NETWORK_NAME)
logger.set_log(basic_info,
               t=gl.get_value('time'),
               filename=FILENAME,
               column=COLUMN,
               affect_days=AFFECT,
               network=net,
               plot_name=PLOT_NAME,
               MSELoss=evaluator.MSELoss(),
               DA=evaluator.DA(),
               Theil=evaluator.Theil(),
               L1Loss=evaluator.L1Loss()
               )
