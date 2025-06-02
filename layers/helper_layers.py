import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import math


class SequenceMaskLayer(nn.Module):

    def __init__(self):
        super(SequenceMaskLayer, self).__init__()

    def forward(self, lengths, maxlen=None, dtype=torch.bool):
        return self.sequence_mask(lengths, maxlen, dtype=dtype)

    @staticmethod
    def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()

        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask = mask.type(dtype)

        return mask


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = rearrange(x, 'b (h1 h2) c -> b c h1 h2', h1 = 7, h2 = 7)
        # b, c, d = x.size()
        # h = np.sqrt(c)
        # x = x.reshape(b, d, h, h)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out = rearrange(out, 'b c h1 h2 -> b (h1 h2) c', h1 = 7, h2 = 7)
        # out = out.reshape()
        return out


class Bottleneck(nn.Module):
    def __init__(self, embed_dim, mid_embed_dim):
        super(Bottleneck, self).__init__()
        self.onehot_predictor = nn.Linear(embed_dim, mid_embed_dim)
        self.onehot_projector = nn.Sequential(
            nn.Linear(mid_embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.Dropout(0.3),
        )
    
    def forward(self, x):
        mid_x = self.onehot_predicter(x)
        x = self.onehot_projector(mid_x)
        return nn.Sigmoid(mid_x), x




class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ratio = 0.2
        out = ratio * self.fc(x) + (1-ratio) * x
        return out


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            # 从均匀分布中抽样数值进行填充
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x



if __name__ == "__main__":
    layer = SELayer(512)
    pooler = Pooler()
    x = torch.rand(12, 49, 512)
    out = layer(x)
    print(out.shape)