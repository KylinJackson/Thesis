import datetime
from torch.utils.data import Dataset
import pandas as pd


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class Action:
    @staticmethod
    def generate_df_affect_by_n_days(series, n, index=False):
        if len(series) <= n:
            raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
        df = pd.DataFrame()
        for i in range(n):
            df['c%d' % i] = series.tolist()[i:-(n - i)]
        df['y'] = series.tolist()[n:]
        if index:
            df.index = series.index[n:]
        return df

    @staticmethod
    def read_data(column='high', n=30, all_too=True, index=False, train_end=-300):
        df = pd.read_csv("上证指数2005-2015.csv", index_col=0)
        df.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), df.index))
        df_column = df[column].copy()
        df_column_train, df_column_test = df_column[:train_end], df_column[train_end - n:]
        df_generate_from_df_column_train = Action.generate_df_affect_by_n_days(df_column_train, n, index=index)
        if all_too:
            return df_generate_from_df_column_train, df_column, df.index.tolist()
        return df_generate_from_df_column_train
