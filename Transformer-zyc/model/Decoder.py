from .utils import MultiHeadAttention, LayerNorm, FeedForwardNetwork, Embedder, PositionEmbedding
import torch.nn as nn
import copy

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_1 = LayerNorm(args)
        self.norm_2 = LayerNorm(args)
        self.norm_3 = LayerNorm(args)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.dropout_2 = nn.Dropout(args.dropout)
        self.dropout_3 = nn.Dropout(args.dropout)
        self.Masked_MHA = MultiHeadAttention(args)
        self.Cross_MHA = MultiHeadAttention(args)
        self.FFN = FeedForwardNetwork(args)

    def forward(self, x, encoder_output, src_mask, tgt_mask): # why dropout(dropout)?
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.Masked_MHA(x1, x1, x1, tgt_mask)) # 自己注意自己，但是不能看到后面的东西，所以是tgt_mask
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.Cross_MHA(x2, encoder_output, encoder_output, src_mask)) # 当前时刻output要去'注意'encoder里的句子，所以是src_mask
        x3 = self.norm_3(x)
        x = x+ self.dropout_3(self.FFN(x3))
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_decoder = args.n_decoder
        self.embed = Embedder(args)
        self.pe = PositionEmbedding(args)
        self.layer = nn.ModuleList([copy.deepcopy(DecoderLayer(args)) for i in range(self.n_decoder)])
        self.norm = LayerNorm(args)

    def forward(self, encoder_output, tgt, src_mask, tgt_mask):
        x = self.embed(tgt)
        x = self.pe(x)
        for i in range(self.n_decoder):
            x = self.layer[i](x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)