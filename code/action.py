import time

import test
import train

TRAIN = True
TEST = True

# 超参数设置
NETWORK = 'LSTM'
AFFECT = 30
STOCK_NAME = '白云机场2011-2015'
TEST_FILENAME = STOCK_NAME + '_test.csv'
TRAIN_FILENAME = STOCK_NAME + '_train.csv'
TITLE = STOCK_NAME
TRAIN_BATCH = 72
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
