import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class Action:
    @staticmethod
    def generate_df(filename, column, index_col, affect):
        # 从csv文件读取数据
        df = pd.read_csv('data/{}'.format(filename), index_col=index_col)

        # 把时间作为index
        df.index = list(
            map(
                lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'),
                df.index)
        )

        # 取出要操作的列
        df_all = df[column].copy()

        # 归一
        mean = df_all.mean()
        std = df_all.std()
        df_standardized = (df_all - mean) / std

        # 重新调整df结构
        df_input = pd.DataFrame()
        for i in range(affect):
            df_input[i] = list(df_standardized)[i:-(affect - i)]
        df_input['y'] = list(df_standardized)[affect:]
        df_input.index = df.index[affect:]
        df_index = list(df_input.index)

        # 生成数据集合
        # cuda0 = torch.device('cuda:0')
        # data = DataSet(torch.Tensor(np.array(df_input), device=cuda0))
        data = DataSet(torch.Tensor(np.array(df_input)).cuda())
        return {
            'dataset': data,
            'real_data': df_all,
            'mean': mean,
            'std': std,
            'index': df_index
        }


if __name__ == '__main__':
    data = Action.generate_df('data_train.csv',
                              'HiPr',
                              'TrdDt',
                              30)
    f_out = open('log/test.txt', 'w')
    print(data['dataset'],
          data['real_data'],
          data['mean'],
          data['std'],
          data['index'],
          sep='\n---\n',
          file=f_out)
    f_out.close()
