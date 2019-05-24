import numpy as np


def data_name(stock_name):
    return '../predict_data/' + stock_name + '_predict_data.npy', \
           '../predict_data/' + stock_name + '_true_data.npy'


def standardize(stock_name):
    predict_data = np.load(data_name(stock_name)[0])
    true_data = np.load(data_name(stock_name)[1])

    mean = true_data.mean()
    std = true_data.std()
    true_std = (true_data - mean) / std
    predict_std = (predict_data - mean) / std
    return true_std, predict_std


stock = '南京化纤'


def r(y1, y2):
    y_true_np = np.array(y1)
    y_predict_np = np.array(y2)
    mean_true = y_true_np.mean()
    mean_predict = y_predict_np.mean()
    numerator = 0.
    denominator = 0.
    d1 = 0.
    d2 = 0.
    for i in range(len(y1)):
        numerator += (y1[i] - mean_true) * (y2[i] - mean_predict)
        d1 += (y1[i] - mean_true) ** 2
        d2 += (y2[i] - mean_predict) ** 2
        denominator = d1 * d2
    denominator = denominator ** (1 / 2)
    return numerator / denominator


print(r(standardize(stock)[0], standardize(stock)[1]))


def mape(y1, y2):
    numerator = 0.
    for i in range(len(y1)):
        unit = y1[i] - y2[i]
        unit /= y1[i]
        numerator += abs(unit)
    return numerator / len(y1)


def MSE(y1, y2):
    numerator = 0.
    for i in range(len(y1)):
        numerator += (y1[i] - y2[i]) ** 2
    numerator /= len(y1)
    return numerator


def l1loss(y1, y2):
    numerator = 0.
    for i in range(len(y1)):
        numerator += abs(y1[i] - y2[i])
    numerator /= len(y1)
    return float(numerator)


def theil(y1, y2):
    numerator = MSE(y1, y2) ** (1 / 2)
    num1 = 0.
    num2 = 0.
    for i in range(len(y1)):
        num1 += y1[i] ** 2
        num2 += y2[i] ** 2
    num1 /= len(y1)
    num2 /= len(y1)
    num1 = num1 ** (1 / 2)
    num2 = num2 ** (1 / 2)
    denominator = num1 + num2
    return numerator / denominator
