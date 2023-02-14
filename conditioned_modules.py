import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, hs, W, ws).contiguous().view(B, C, H*hs, W*ws)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, (H//hs)*(W//ws), hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x