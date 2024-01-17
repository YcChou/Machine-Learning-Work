from .utils import MultiHeadAttention, LayerNorm, FeedForwardNetwork, Embedder, PositionEmbedding
import torch.nn as nn
import copy

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_1 = LayerNorm(args)
        self.norm_2 = LayerNorm(args)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.dropout_2 = nn.Dropout(args.dropout)
        self.FFN = FeedForwardNetwork(args)
        self.MHA = MultiHeadAttention(args)

    def forward(self, x, src_mask):
        x1 = self.norm_1(x) # why norm firstly?
        x = x + self.dropout_1(self.MHA(x1, x1, x1, src_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.FFN(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_encoder = args.n_encoder
        self.embed = Embedder(args)
        self.pe = PositionEmbedding(args)
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(args)) for i in range(args.n_encoder)])
        self.norm_last = LayerNorm(args)

    def forward(self, src, src_mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.n_encoder):
            x = self.layers[i](x, src_mask)
        return self.norm_last(x) # why norm_last?
