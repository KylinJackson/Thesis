import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import globalvar as gl
from data import Action
from network import *
from record import Logger

# 超参数设置
NETWORK_NAME = gl.get_value('network')
AFFECT = gl.get_value('affect')
HIDDEN_SIZE = gl.get_value('hidden_size')
NUM_LAYERS = gl.get_value('num_layers')
LR = gl.get_value('lr')
EPOCH = gl.get_value('epoch')
FILENAME = gl.get_value('train_filename')
COLUMN = gl.get_value('column')
INDEX_COL = 'TrdDt'
BATCH_SIZE = gl.get_value('train_batch')

# 加载数据
data = Action.generate_df(
    FILENAME,
    COLUMN,
    INDEX_COL,
    AFFECT
)
data_loader = DataLoader(data['dataset'], batch_size=BATCH_SIZE, shuffle=False)

# 生成网络
net = eval(NETWORK_NAME)(AFFECT, HIDDEN_SIZE, NUM_LAYERS)
optimizer = optim.Adam(net.parameters(), lr=LR)
loss_func = nn.MSELoss()

for step in range(EPOCH):
    loss = None
    for tx, ty in data_loader:
        output = net(tx.reshape(1, BATCH_SIZE, AFFECT))
        loss = loss_func(output.reshape(BATCH_SIZE), ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('step: {}, loss: {}'.format(step + 1, loss.detach()))
torch.save(net, 'save/{}.pt'.format(NETWORK_NAME))

# 生成日志
logger = Logger('train.log')
basic_info = 'trained {} for {} times.'.format(NETWORK_NAME, EPOCH)
logger.set_log(basic_info,
               filename=FILENAME,
               column=COLUMN,
               affect_days=AFFECT,
               learning_rate=LR,
               network=net,
               save_name='{}.pt'.format(NETWORK_NAME)
               )
