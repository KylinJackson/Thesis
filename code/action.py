import time

import test
import train

TRAIN = False
TEST = True

# 超参数设置
NETWORK = 'LSTM'
AFFECT = 30
TEST_FILENAME = '华塑控股_test.csv'
TRAIN_FILENAME = '华塑控股_train.csv'
TITLE = '华塑控股'
TRAIN_BATCH = 20
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
