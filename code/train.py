import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Action
from network import *
from record import Logger


def get_value_or_default(kwargs, key, default=None):
    try:
        return kwargs[key]
    except KeyError:
        return default


def train(train_filename, **kwargs):
    # 超参数设置
    network_name = get_value_or_default(kwargs, 'network', 'LSTM')
    affect = get_value_or_default(kwargs, 'affect', 30)
    lr = get_value_or_default(kwargs, 'lr', 0.0001)
    epoch = get_value_or_default(kwargs, 'epoch', 2000)
    filename = train_filename
    column = get_value_or_default(kwargs, 'column', 'ClPr')
    index_col = 'TrdDt'
    batch_size = get_value_or_default(kwargs, 'train_batch', 1)

    # 加载数据
    data = Action.generate_df(
        filename,
        column,
        index_col,
        affect
    )
    data_loader = DataLoader(data['dataset'], batch_size=batch_size, shuffle=False)

    # 生成网络
    # net = LSTM(affect).cuda()
    net = eval(network_name)(affect).cuda()
    optimizer = optim.Adam(net.parameters())
    loss_func = nn.MSELoss().cuda()

    for step in range(epoch):
        loss = None
        for tx, ty in data_loader:
            output = net(tx.reshape(1, batch_size, affect))
            loss = loss_func(output.reshape(batch_size), ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('step: {}, loss: {}'.format(step + 1, loss.detach()))
    torch.save(net, 'save/{}.pt'.format(network_name))

    # 生成日志
    logger = Logger('train.log')
    basic_info = 'trained {} for {} times.'.format(network_name, epoch)
    logger.set_log(basic_info,
                   filename=filename,
                   column=column,
                   affect_days=affect,
                   learning_rate=lr,
                   network=net,
                   save_name='{}.pt'.format(network_name)
                   )
