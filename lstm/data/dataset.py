import os.path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ETTh1Dataset(Dataset):
    def __init__(self, config, split="train"):
        # 读取CSV文件并存储数据
        file_path = os.path.join(config.data_path, split+"_set.csv")
        df = pd.read_csv(file_path, index_col=0)
        scaler = MinMaxScaler()
        scaler_model = MinMaxScaler()
        self.data = scaler_model.fit_transform(np.array(df))
        scaler.fit_transform(np.array(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']]).reshape(-1, 1))

        self.input_size = config.input_length
        self.output_size = config.output_length

        self.data_x = []
        self.data_y = []
        self.split_data()

    # 形成训练数据
    def split_data(self):
        dataX = []  # 保存X
        dataY = []  # 保存Y

        # 将输入窗口的数据保存到X中，将输出窗口保存到Y中
        window_size = self.input_size + self.output_size
        for index in range(len(self.data) - window_size):
            dataX.append(self.data[index: index + self.input_size][:])
            dataY.append(self.data[index + self.input_size: index + window_size][:])

        self.data_x = np.array(dataX)
        self.data_y = np.array(dataY)

    def __len__(self):
        # 返回数据的总数
        return len(self.data_x)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_x[idx])
        label = torch.tensor(self.data_y[idx])
        return data, label


