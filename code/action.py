import time

import globalvar as gl

TRAIN = True
TEST = True

# 超参数设置
gl.set_value('network', 'LSTM')
gl.set_value('affect', 30)
gl.set_value('test_filename', '永安林业_test.csv')
gl.set_value('train_filename', '永安林业_train.csv')
gl.set_value('title', '永安林业')
gl.set_value('train_batch', 10)
gl.set_value('column', 'Clpr')
gl.set_value('time', time.localtime())
# gl.set_value('hidden_size', 64)
# gl.set_value('num_layers', 1)
gl.set_value('lr', 0.0001)
gl.set_value('epoch', 2000)

if TRAIN:
    import train
if TEST:
    import test
