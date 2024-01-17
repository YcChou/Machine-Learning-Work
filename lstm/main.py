import os.path
import random

import yaml
import argparse
import torch
from easydict import EasyDict
from utils.utils import set_seed
from data import create_data_loaders
from model.LSTM import LSTM
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main(config):
    for key, value in config.items():
        print(f'{key}: {value}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_dataloader, test_loader = create_data_loaders(config)
    model = LSTM(config).to(device)

    if config.loss_type.lower() == 'mae':
        criterion = torch.nn.L1Loss().to(device)
    elif config.loss_type.lower() == 'mse':
        criterion = torch.nn.MSELoss().to(device)
    else:
        raise RuntimeError("loss type error. Only accept mae or mse")
    # criterion_mae = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 调用train函数进行训练
    train(model, train_loader, val_dataloader, optimizer, criterion, device, config)
    test_loss = test(model, test_loader, criterion, device, config)
    print("test loss: {}".format(test_loss))
    draw(model, test_loader, config.save_path, device)


def train(model, train_loader, val_loader, optimizer, criterion, device, config):
    for epoch in range(config.epoch):
        model.train()  # 设置模型为训练模式
        for batch in tqdm(train_loader):
            # 获取输入数据和标签
            inputs, labels = batch
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # 在验证集上评估模型
        val_loss = test(model, val_loader, criterion, device, config)

        # 打印当前Epoch的训练和验证损失等信息
        print(f"Epoch {epoch + 1}/{config.epoch}, Train Loss: {loss.item()}, "
              f"Val Loss: {val_loss.item()}")


def test(model, loader, criterion, device, config):
    model.eval()  # 设置模型为评估模式
    loss = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels)
    return loss


def draw(model, test_loader, save_path, device):
    model.eval()
    title = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    with torch.no_grad():
        for batch in test_loader:
            rand_index = random.randint(0, len(batch) - 1)
            input_r, label_r = batch
            input_r = input_r.float().to(device)
            label_r = label_r.float().to(device)
            output = model(input_r)
            input_r = input_r[rand_index].squeeze(0).cpu().numpy()
            label_r = label_r[rand_index].squeeze(0).cpu().numpy()
            output = output[rand_index].squeeze(0).cpu().numpy()

            fig = plt.figure(figsize=(15, 6))
            for feature_index in range(input_r.shape[1]):
                # 提取当前特征的数据
                current_feature = input_r[:, feature_index]
                current_label = label_r[:, feature_index]
                current_feature = np.concatenate((current_feature, current_label), axis=0)
                current_output = output[:, feature_index]

                # 创建一个新的子图
                ax = fig.add_subplot(3, 3, feature_index + 1)

                # 绘制这个特征随时间变化的曲线
                plt.sca(ax)
                plt.plot(range(1, len(current_feature)+1), current_feature, label='ground_truth')
                plt.plot(range(97, 97+len(current_output)), current_output, label='prediction')

                # 设置标题和其他属性
                plt.title(title[feature_index])
                plt.xlabel('Time Step')
                plt.ylabel('Feature Value')
                plt.legend()

            # 显示所有的子图
            plt.tight_layout()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = os.path.join(save_path, 'pred.png')
            plt.savefig(filename, dpi=300)

            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args = parser.parse_args()
    cfg = EasyDict(yaml.load(open(args.config, 'r'), Loader=yaml.Loader))

    set_seed(cfg)
    main(cfg)
