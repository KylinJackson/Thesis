import time

import test
import train

TRAIN = True
TEST = True

run_list_dict = {
    '上证指数': '2009-2014',
    '深证成指': '2011-2015',
    '东风汽车': '2011-2014',
    '中信证券': '2011-2014',
    '浦发银行': '2011-2015',
    '武钢股份': '2011-2014',
    '中国石化': '2011-2014',
    '中国平安': '2011-2014',
}

for key in run_list_dict:
    print(key, run_list_dict[key])
    # 超参数设置
    NETWORK = 'LSTM'
    AFFECT = 30
    STOCK_NAME = key
    TIME = run_list_dict[key]
    TITLE = STOCK_NAME
    STOCK_NAME += TIME
    TEST_FILENAME = STOCK_NAME + '_test.csv'
    TRAIN_FILENAME = STOCK_NAME + '_train.csv'
    TRAIN_BATCH = 10
    COLUMN = 'ClPr'
    TIME_NOW = time.localtime()
    LR = 0.0001
    EPOCH = 2000

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
        test.test(TEST_FILENAME,
                  TIME_NOW,
                  TITLE,
                  network=NETWORK,
                  affect=AFFECT,
                  column=COLUMN)
