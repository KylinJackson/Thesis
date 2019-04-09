import datetime

import pandas as pd
from torch.utils.data import Dataset


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class Action:
    @staticmethod
    def generate_df(filename, column, index_col, affect, train_end):
        df = pd.read_csv('data/{}'.format(filename), index_col=index_col)
        df.index = list(
            map(
                lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'),
                df.index)
        )
        df_all = df[column].copy()
        # generate train data
        df_column_train, df_column_test = df_all[:train_end], df_all[train_end - affect:]
        df_train = pd.DataFrame()
        for i in range(affect):
            df_train['c%d' % i] = df_column_train.tolist()[i:-(affect - i)]
        df_train['y'] = df_column_train.tolist()[affect:]
        return df_train, df_all, df.index.tolist()
