import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
# DDP
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
# from custom files
from utils.LoadData import dataloader
from model.Transformer import Transformer
from log import init_logger
from utils.mask import create_masks
import numpy as np
import matplotlib.pyplot as plt

Huggingface_model_cache = os.environ.get('TRANSFORMERS_CACHE')

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer')
    # Transformer model parameter
    parser.add_argument('--d_model', default=512, type=int, help='the dimension of the model')
    parser.add_argument('--n_head', default=8, type=int, help='the number of heads in the multi-head self-attention module')
    parser.add_argument('--n_encoder', default=6, type=int, help='the number of encoder layers')
    parser.add_argument('--n_decoder', default=6, type=int, help='the number of decoder layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='the radio of dropout')
    parser.add_argument('--eps', default=1e-6, type=float, help='')
    parser.add_argument('--d_ff', default=2048, type=int, help='the hidden vector size in middle layer')

    # Input size
    parser.add_argument('--input_size', default=96, type=int, required=True)
    parser.add_argument('--output_size', default=96, type=int, required=True)

    # train
    parser.add_argument('--epochs', default=3, type=int, help='the epochs you want to train')
    parser.add_argument('--print_every', default=100, type=int, help='every n iters training logs are generated ', required=True)
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=4, type=int, required=True)
    parser.add_argument('--data_path', default=None, type=str, help='use a space to separate if there is multi datasets', required=True)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    # parser.add_argument('--weight_decay', default=True, type=bool, help='?')
    # others
    parser.add_argument('--log_file', default='', type=str, help='file path of logging', required=True)
    parser.add_argument('--save_path', default='', type=str, help='file path of saving', required=True)
    parser.add_argument('--mode', default='', type=str, help='wanna to train, valid or test the model', choices=['train', 'dev', 'test'], required=True)
    parser.add_argument('--nodes', default=2, type=int, help='the number of gpus used')
    parser.add_argument('--gpu', default=0, type=int, help='single gpu for valid and test')
    parser.add_argument('--special_tokens', action='store_true')
    parser.add_argument('--figure_path', type=str, help='the path to save figures')


    args = parser.parse_args()
    return args


def train(args, model):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log = init_logger(args.log_file, mode='w')
    # use DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl") # 初始化
    torch.cuda.set_device(local_rank) # 当前显卡
    print(f'Use GPU: {local_rank} for {args.mode}ing')
    args.nodes = torch.distributed.get_world_size()
    rank = dist.get_rank()

    # load train data
    train_dataloader = dataloader(args)
    
    total_items = len(train_dataloader) * args.batch_size * args.nodes * args.epochs # len(train_dataloader)获取的是batch数量，也就是装载器装了多少个batch
    # total_items = len(train_dataloader) * args.batch_size * args.epochs # len(train_dataloader)获取的是batch数量，也就是装载器装了多少个batch
    total_steps = total_items // (args.nodes * args.batch_size)

    if rank == 0:
        log.info(f'{total_items} number of items will be fed into the model.')
        log.info(f'nodes:{args.nodes}, batch_size:{args.batch_size}, epochs:{args.epochs}, total_steps:{total_steps}')
        log.info(' An input example is given as follows:')
        log.info(next(iter(train_dataloader)))

    model = model.cuda() # 如果用DDP，要在配置DDP之后to cuda
    # initialize
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
        output_device=local_rank, find_unused_parameters=True) # Data Parallel
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # train
    model.train() # why use this?
    start = time.time()
    temp = start
    total_loss = 0

    i = 0
    for epoch in range(args.epochs):
        for inputs, tgt_input, label in train_dataloader:

            src = inputs # [bs, 96, 7]

            targets = label.contiguous().view((-1, 7)) # [bs*96, 7]

            # tgt_intput: <CLS> + sentence, [bs, m-1]
            # targets   : sentence + <SEP>, [bs, m-1] --> (m-1)*bs，行向量
            
            src_mask, trg_mask = create_masks(torch.ones(args.batch_size,args.input_size), torch.ones(args.batch_size,args.output_size))
            # import pdb; pdb.set_trace()

            optim.zero_grad()
            preds = model(src, tgt_input, src_mask, trg_mask) # [bs, n, 7]

            # out = F.softmax(preds, dim=-1)
            # val, idx = out[:, 0].data.topk(1)
            # print(idx)
            # import pdb; pdb.set_trace()
            loss = F.mse_loss(preds, label.cuda())
            # loss = F.mse_loss(preds.view(-1, 7), targets.cuda())
            # loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets.cuda()) # cross_entropy([bs*len, 7], [bs*len, 7])
            loss.backward()
            optim.step()
            total_loss += loss.item()
            i += 1

            if (i + 1) % args.print_every == 0:
                loss_avg = total_loss / args.print_every
                step = (i + 1)
                if rank == 0:
                    log.info(f'time = {int((time.time() - start) // 60)}m, epoch {epoch + 1}, step = {step}, loss = {loss_avg:.5f},'
                                f'progress:{step*100/total_steps:.1f}%, {time.time() - temp:.2f}s per {args.print_every} batches')
                    
                    # out = F.softmax(preds, dim=-1)
                    # val, idx = out[:, 2].data.topk(1)
                    # log.info(idx)
                torch.save(model.module.state_dict(), args.save_path + '/' + str(step) + '.pth')
                total_loss = 0
                temp = time.time()

def valid_single(args, dev_dataloader, model, pth_path, log):
    device = torch.device("cuda:{}".format(args.gpu))
    try:
        model.load_state_dict(torch.load(pth_path), strict=True) # 先DDP，再cuda，再装权重
    except:
        model.load_state_dict(torch.load(pth_path, map_location=device), strict=True) # 先DDP，再cuda，再装权重

    model.to(device)
    model.eval()

    log.info(f"Loaded model from {pth_path}")

    start = time.time()
    total_loss = 0
    
    for inputs, tgt_input, label in dev_dataloader:
        src = inputs.cuda()
        targets = label.contiguous().view((-1, 7)) # [bs*len, 7]

        src_mask, tgt_mask = create_masks(torch.ones(args.batch_size,args.input_size), torch.ones(args.batch_size,args.output_size))

        outputs = torch.zeros((args.batch_size, args.output_size, 7))
        outputs = outputs.to(torch.float32).cuda() # [bs, n, 7]

        # 解码过程
        for t in range(args.output_size): # 解码到了t时刻
            with torch.no_grad():
                out = model(src, outputs, src_mask, tgt_mask) # [bs, n, 7]
                pred = out[:, t, :] # [bs, 7] 每一行都是这个bs当前t时刻的预测

                if t == args.output_size-1:
                    outputs = torch.cat([outputs, torch.zeros(args.batch_size, 1, 7).cuda()], dim=1)

                outputs[:,t+1,:]=pred

        outputs = outputs[:, 1:, :]
        loss = F.mse_loss(outputs.reshape(-1, outputs.size(-1)), targets.cuda())

        # import pdb; pdb.set_trace()
        # loss = F.cross_entropy(out.view(-1, out.size(-1)), targets, ignore_index=0)
        total_loss += loss.item()

        # log.info(f'time = {int((time.time() - start) // 60)}m, loss = {loss:.3f}')

    loss_avg = total_loss / len(dev_dataloader)
    log.info(f'time = {int((time.time() - start) // 60)}m, ave_loss = {loss_avg:.3f}')
    return

def valid(args, model):
    log = init_logger(args.log_file, mode='w')

    device = torch.device("cuda:{}".format(args.gpu))
    print(f'Use GPU: {device} for {args.mode}ing')

    # load valid data
    dev_dataloader = dataloader(args)

    log.info(f'{len(dev_dataloader) * args.batch_size} number of items will be valid by models.')
    log.info(f'node_id:{device}, batch_size:{args.batch_size}')

    model_files = [f for f in os.listdir(args.save_path) if f.endswith('.pth')]
    # 循环加载模型
    for model_file in model_files:
        pth_path = os.path.join(args.save_path, model_file)
        valid_single(args, dev_dataloader, model, pth_path, log)

    return

def test(args, model):
    log = init_logger(args.log_file, mode='w')

    device = torch.device("cuda:{}".format(args.gpu))
    print(f'Use GPU: {device} for {args.mode}ing')

    # load test data
    test_dataloader = dataloader(args)
    log.info(f'{len(test_dataloader) * args.batch_size} number of items will be tested by models.')
    log.info(f'node_id:{device}, batch_size:{args.batch_size}')

    model.load_state_dict(torch.load(args.save_path), strict=True) # 先DDP，再cuda，再装权重
    model.to(device)
    model.eval()

    log.info(f"Loaded model from {args.save_path}")

    start = time.time()

    if_draw = True

    for inputs, tgt_input, label in test_dataloader:
        total_loss = {'mse':0, 'mae':0, 'std':0}
        src = inputs.cuda()
        targets = label.contiguous().view((-1, label.shape[-1])) # [bs*len, 7]

        src_mask, tgt_mask = create_masks(torch.ones(args.batch_size,args.input_size), torch.ones(args.batch_size,args.output_size))

        outputs = torch.zeros((args.batch_size, args.output_size, 7))
        outputs = outputs.to(torch.float32).cuda() # [bs, n, 7]

        # 解码过程
        for t in range(args.output_size): # 解码到了t时刻
            with torch.no_grad():
                out = model(src, outputs, src_mask, tgt_mask) # [bs, n, 7]
                pred = out[:, t, :] # [bs, 7] 每一行都是这个bs当前t时刻的预测

                if t == args.output_size-1:
                    outputs = torch.cat([outputs, torch.zeros(args.batch_size, 1, 7).cuda()], dim=1)

                outputs[:,t+1,:]=pred
                # import pdb; pdb.set_trace()
        outputs = outputs[:, 1:, :]
        mse_loss = F.mse_loss(outputs, label.cuda())
        mae_loss = F.l1_loss(outputs, label.cuda())
        import pdb; pdb.set_trace()

        # 画图
        if if_draw:
            if_draw = draw(args, outputs.permute(0, 2, 1).cpu().numpy(), torch.cat([inputs, label], dim=1).permute(0, 2, 1).cpu().numpy())

        # loss = F.cross_entropy(out.view(-1, out.size(-1)), targets, ignore_index=0)
        total_loss['mse'] += mse_loss.item()
        total_loss['mae'] += mae_loss.item()
        print(total_loss['mse'], total_loss['mae'])

        # log.info(f'time = {int((time.time() - start) // 60)}m, loss = {loss:.3f}')

    # loss_avg = total_loss / len(test_dataloader)
    # log.info(f'time = {int((time.time() - start) // 60)}m, loss = {loss_avg:.3f}')
    return

def draw(args, npList, label):
    import random
    index = random.randint(0, args.batch_size-1)

    prediction_list = npList[index]
    label_list = label[index]

    fig = plt.figure(figsize=(15, 6))
    legend = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    for i in range(7):
        # 创建一个新的子图
        if i == 6:
            ax = fig.add_subplot(3, 1, 3)
        else:
            ax = fig.add_subplot(3, 3, i + 1)

        # 绘制这个特征随时间变化的曲线
        plt.sca(ax)
        x = np.arange(0, args.input_size+args.output_size)

        # import pdb; pdb.set_trace()
        plt.plot(x, label_list[i], label='Ground Truth')
        plt.plot(x[args.input_size:], prediction_list[i], label='Prediction')

        # 添加标签和图例
        plt.title(legend[i])
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(args.figure_path+'/features.png')  # 文件格式通过文件名的后缀指定


    # 画OT
    plt.clf()
    x = np.arange(0, args.input_size+args.output_size)

    plt.plot(x, label_list[i], label='Ground Truth')
    plt.plot(x[args.input_size:], prediction_list[i], label='Prediction')

    # 添加标签和图例
    plt.title(legend[i])
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.figure_path+'/OT.png')
    return False

def main():
    args = parse_args()
    
    # initialize and configure the DDP
    if args.mode == 'train':
        model = Transformer(args)
        train(args, model)

    if args.mode == 'dev':
        model = Transformer(args)
        valid(args, model)

    if args.mode == 'test':
        model = Transformer(args)
        test(args, model)

main()

# Dataloader bs = 64, node = 4, bs_all = 64 * 4, 每个node都是64的bs
# 过一个batch就是一个step，在ddp中，过batch*node，就是一个step