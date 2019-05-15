import numpy as np
import os


class Evaluate:
    class LenError(RuntimeError):
        def __init__(self):
            print('长度不一致')

    def __init__(self, title, y_true, y_predict):
        if not isinstance(y_true, list):
            self.y_true = y_true.tolist()
        else:
            self.y_true = y_true
        if not isinstance(y_predict, list):
            self.y_predict = y_predict.tolist()
        else:
            self.y_predict = y_predict
        self.check()
        if not os.path.exists('predict_data'):
            os.mkdir('predict_data')
        else:
            np.save('predict_data/{}_true_data.npy'.format(title), np.array(self.y_true))
            np.save('predict_data/{}_predict_data.npy'.format(title), np.array(self.y_predict))

    def check(self):
        if len(self.y_true) != len(self.y_predict):
            raise self.LenError()

    def MSELoss(self):
        self.check()
        numerator = 0.
        for i in range(len(self.y_true)):
            numerator += (self.y_true[i] - self.y_predict[i]) ** 2
        numerator /= len(self.y_true)
        return numerator

    def Theil_U(self, ):
        self.check()
        numerator = self.MSELoss() ** (1 / 2)
        num1 = 0.
        num2 = 0.
        for i in range(len(self.y_true)):
            num1 += self.y_true[i] ** 2
            num2 += self.y_predict[i] ** 2
        num1 /= len(self.y_true)
        num2 /= len(self.y_true)
        num1 = num1 ** (1 / 2)
        num2 = num2 ** (1 / 2)
        denominator = num1 + num2
        return numerator / denominator

    def DA(self):
        self.check()

        da = 0

        for i in range(1, len(self.y_true)):
            if self.y_true[i] > self.y_true[i - 1] and self.y_predict[i] > self.y_true[i - 1]:
                da += 1
            elif self.y_true[i] == self.y_true[i - 1] and -0.01 < self.y_predict[i] - self.y_true[i - 1] < 0.01:
                da += 1
            elif self.y_true[i] < self.y_true[i - 1] and self.y_predict[i] < self.y_true[i - 1]:
                da += 1
        return da / (len(self.y_true) - 1)

    def L2Loss(self):
        return self.MSELoss()

    def L1Loss(self):
        self.check()
        numerator = 0.
        for i in range(len(self.y_true)):
            numerator += abs(self.y_true[i] - self.y_predict[i])
        numerator /= len(self.y_true)
        return float(numerator)

    def R(self):
        self.check()
        y_true_np = np.array(self.y_true)
        y_predict_np = np.array(self.y_predict)
        mean_true = y_true_np.mean()
        mean_predict = y_predict_np.mean()
        numerator = 0.
        denominator = 0.
        for i in range(len(self.y_true)):
            numerator += (self.y_true[i] - mean_true) * (self.y_predict[i] - mean_predict)
            denominator += (self.y_true[i] - mean_true) ** 2 * (self.y_predict[i] - mean_predict) ** 2
        denominator = denominator ** (1 / 2)
        return numerator / denominator

    def MAPE(self):
        self.check()
        self.check()
        numerator = 0.
        for i in range(len(self.y_true)):
            unit = self.y_true[i] - self.y_predict[i]
            unit /= self.y_true[i]
            numerator += unit
        return numerator / len(self.y_true)

    def customize(self):
        self.check()
        score = 0
        for i in range(1, len(self.y_true)):
            rate_true = (self.y_true[i - 1] - self.y_true[i]) / self.y_true[i - 1]
            rate_predict = (self.y_predict[i - 1] - self.y_predict[i]) / self.y_predict[i - 1]
            if 0.06 <= rate_true <= 0.1:
                if 0.06 <= rate_predict <= 0.1:
                    score += 2
                if 0. <= rate_predict <= 0.06:
                    score += 1
                if -0.06 <= rate_predict <= 0.:
                    pass
                if -0.1 <= rate_predict <= -0.06:
                    pass
            if 0. <= rate_true <= 0.06:
                if 0.06 <= rate_predict <= 0.1:
                    score += 1
                if 0. <= rate_predict <= 0.06:
                    score += 1
                if -0.06 <= rate_predict <= 0.:
                    pass
                if -0.1 <= rate_predict <= -0.06:
                    pass
            if -0.06 <= rate_true <= 0.:
                if 0.06 <= rate_predict <= 0.1:
                    score -= 1
                if 0. <= rate_predict <= 0.06:
                    score -= 1
                if -0.06 <= rate_predict <= 0.:
                    score += 1
                if -0.1 <= rate_predict <= -0.06:
                    score += 1
            if -0.1 <= rate_true <= -0.06:
                if 0.06 <= rate_predict <= 0.1:
                    score -= 2
                if 0. <= rate_predict <= 0.06:
                    score -= 1
                if -0.06 <= rate_predict <= 0.:
                    score += 1
                if -0.1 <= rate_predict <= -0.06:
                    score += 2
        return score
