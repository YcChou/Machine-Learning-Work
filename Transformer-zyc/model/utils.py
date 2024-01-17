import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.heads = args.n_head
        self.d_k = self.d_model // self.heads

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.droptout = nn.Dropout(args.dropout)
        self.out = nn.Linear(self.d_model, self.d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # (bs, h, -1, d_k) * (bs, h, d_k, -1) = （bs, h, n, n）

        if mask is not None: # mask: (bs, n, n)
            mask = mask.unsqueeze(1) # 在头维度进行一样的mask操作
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores) # what is the type of dropout?

        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        # q,k,v (bs, n, d)
        bs = q.size(0)
        # divid to h heads
        q = self.q_linear(q).view(bs, -1, self.heads, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.heads, self.d_k)

        q = q.transpose(1, 2) # (bs, h, -1, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate att
        scores = self.attention(q, k, v, self.d_k, mask, self.droptout)
        # concat heads
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)

        # 当调用contiguous()时，会强制拷贝一份tensor
        # x = torch.randn(3, 2)
        # y = torch.transpose(x, 0, 1).contiguous()
        # 修改x的转置对x没有影响

class LayerNorm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(args.d_model))
        self.bias = nn.Parameter(torch.zeros(args.d_model))
        self.eps = args.eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear_1 = nn.Linear(args.d_model, args.d_ff)
        self.dropout = nn.Dropout(args.dropout)
        self.linear_2 = nn.Linear(args.d_ff, args.d_model)

    def forward(self, x):
        x = self.dropout(self.linear_1(x))
        x = self.linear_2(x)
        return x
    
class PositionEmbedding(nn.Module):
    '''
    parameter: 
    input: sentence after embedding []
    '''
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model

        # Matrix PE is a constant
        pe = torch.zeros(args.output_size, self.d_model)
        for pos in range(args.output_size):
            for i in range(0, self.d_model, 2): # i = 0, 2,...,510
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / self.d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # 随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)

        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x

class Embedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.features = 7
        self.embedding = nn.Linear(self.features, args.d_model)
        # self.embedding = nn.Embedding(self.features, args.d_model)
    
    def forward(self, input_seq):

        return self.embedding(input_seq)
