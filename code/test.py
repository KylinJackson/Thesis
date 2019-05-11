import torch
from torch.utils.data import DataLoader

from data import Action
from evaluate import Evaluate
from plot import Plot
from record import Logger


def get_value_or_default(kwargs, key, default=None):
    try:
        return kwargs[key]
    except KeyError:
        return default


def test(test_filename, time_now, title, **kwargs):
    # 超参数设置
    network_name = get_value_or_default(kwargs, 'network', default='LSTM')
    affect = get_value_or_default(kwargs, 'affect', default=30)
    filename = test_filename
    column = get_value_or_default(kwargs, 'column', default='ClPr')
    index_col = 'TrdDt'
    batch_size = 1
    plot_name = get_value_or_default(kwargs, 'plot_name', default=['fig1', ])

    # 加载数据
    data = Action.generate_df(
        filename,
        column,
        index_col,
        affect
    )
    data_loader = DataLoader(data['dataset'], batch_size=batch_size, shuffle=False)

    net = torch.load('save/{}.pt'.format(network_name))

    predict = list()
    for tx, ty in data_loader:
        output = net(tx.reshape(1, batch_size, affect))
        output = output.reshape(1).detach()
        predict.append(float(output) * data['std'] + data['mean'])

    plt1 = Plot(1, time_now, network_name)
    plt1.plot(data['index'], data['real_data'][affect:], 'real data')
    plt1.plot(data['index'], predict, 'predict data')
    plt1.title(title, zh=True)
    plt1.save(plot_name[0])
    # Plot.show()
    Plot.cla()

    evaluator = Evaluate(title, data['real_data'][affect:], predict)

    logger = Logger('test.log')
    basic_info = 'tested {}.'.format(network_name)
    logger.set_log(basic_info,
                   t=time_now,
                   filename=filename,
                   column=column,
                   affect_days=affect,
                   network=net,
                   plot_name=plot_name,
                   MSELoss=evaluator.MSELoss(),
                   DA=evaluator.DA(),
                   Theil=evaluator.Theil_U(),
                   L1Loss=evaluator.L1Loss(),
                   Customize=evaluator.customize(),
                   title=title,
                   MAPE=evaluator.MAPE(),
                   R=evaluator.R()
                   )
    f_out = open('log/{}.txt'.format(title), 'w')
    print('{} = {}'.format('time', time_now),
          '{} = {}'.format('MSELoss', evaluator.MSELoss()),
          '{} = {}'.format('DA', evaluator.DA()),
          '{} = {}'.format('Theil_U', evaluator.Theil_U()),
          '{} = {}'.format('L1Loss', evaluator.L1Loss()),
          '{} = {}'.format('MAPE', evaluator.MAPE()),
          '{} = {}'.format('R', evaluator.R()),
          file=f_out,
          sep='\n')
    f_out.close()
    return evaluator
