import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ETTh1Dataset(Dataset):
    def __init__(self, args):
        if args.mode == 'train':
            df = pd.read_csv(args.data_path+'/train_set.csv', index_col=0)
        elif args.mode == 'dev':
            df = pd.read_csv(args.data_path+'/validation_set.csv', index_col=0)
        else:
            df = pd.read_csv(args.data_path+'/test_set.csv', index_col=0)

        scaler = MinMaxScaler()
        scaler_model = MinMaxScaler()
        self.data = scaler_model.fit_transform(np.array(df))
        scaler.fit_transform(np.array(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']]).reshape(-1, 1))

        self.input_size = args.input_size
        self.output_size = args.output_size

        self.data_x = []
        self.data_yin = []
        self.data_yout = []
        self.split_data(args)

    # 形成训练数据
    def split_data(self, args):
        dataX = []  # 保存X
        dataY = []  # 保存Y
        dataY_in = []

        # 将输入窗口的数据保存到X中，将输出窗口保存到Y中
        window_size = self.input_size + self.output_size
        # for index in range(0, len(self.data) - window_size, self.input_size):
        for index in range(len(self.data) - window_size):
            dataX.append(self.data[index: index + self.input_size][:])
            dataY.append(self.data[index + self.input_size: index + window_size][:])

        # 重构transformer decoder数据
        SOS = np.zeros((1, 7))
        for i in range(len(dataY)):
            # prefix = dataX[i][-args.special_tokens:, :] # [18, 7]
            # dataY_in.append(np.concatenate((prefix, dataY[i][:-1,:]), axis=0))
            dataY_in.append(np.concatenate((SOS, dataY[i][:-1,:]), axis=0))
            
        dataY_out = dataY
        self.data_x = np.array(dataX)
        self.data_yin = np.array(dataY_in)
        self.data_yout = np.array(dataY_out)

    def __len__(self):
        # 返回数据的总数
        return len(self.data_x)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_x[idx], dtype=torch.float32)
        decoder_input = torch.tensor(self.data_yin[idx], dtype=torch.float32)
        label = torch.tensor(self.data_yout[idx], dtype=torch.float32)
        return data, decoder_input, label

def dataloader(args):

    raw_dataset = ETTh1Dataset(args)

    # def collate_fn(data):
        # '''
        # different datasets need different ways to deal with
        # '''
        # inputs = {}

        # if args.special_tokens:
        #     def construct_tgt_input(tensor):

        #         SOS = torch.zeros(1, 7)
        #         tgt_input = torch.cat((SOS, tensor[:-1,:]), dim=0)
        #         return tgt_input
            
        #     inputs['tgt_input'] = [construct_tgt_input(item[1]) for item in data]
        #     inputs['tgt_output'] = [item[1] for item in data]
        #     inputs['inputs'] = [[item[0] for item in data]]
        # # tgt_intput: <CLS> + sentence, [:-1]
        # # targets   : sentence + <SEP>, [1: ]
        # # 不能在padding过后移位，要在分词时移位，错位就是<CLS>和<SEP>的区别
        # # 如果记分词之后的句子长度，src为n，不错位的tgt为m，那么错位后的tgt_input和targets是m-1
        # # label分为两个，左移和右移作为输入和输出

        # return inputs
    
    # create DataLoader:
    if args.mode == 'train':
        sampler = DistributedSampler(raw_dataset)
        dataloader = DataLoader(raw_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    else:
        dataloader = DataLoader(raw_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True) # 单卡

    return dataloader