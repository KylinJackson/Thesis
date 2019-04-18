import time

import globalvar as gl

TRAIN = False
TEST = False

# 超参数设置
gl.set_value('network', 'LSTM')
gl.set_value('affect', 30)
gl.set_value('test_filename', 'data_test.csv')
gl.set_value('train_filename', 'data_train.csv')
gl.set_value('train_batch', 10)
gl.set_value('column', 'HiPr')
gl.set_value('time', time.localtime())
gl.set_value('hidden_size', 64)
gl.set_value('num_layers', 1)
gl.set_value('lr', 0.0001)
gl.set_value('epoch', 1000)

if TRAIN:
    import train
if TEST:
    import test
