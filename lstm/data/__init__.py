import torch
import numpy as np
from data.dataset import ETTh1Dataset
from torch.utils.data import DataLoader


def create_data_loaders(config):
    # 创建数据集实例
    train_dataset = ETTh1Dataset(config, split='train')
    test_dataset = ETTh1Dataset(config, split='test')
    val_dataset = ETTh1Dataset(config, split='validation')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader
