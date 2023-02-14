import torch.nn as nn
import torch
import torch.nn.functional as F

# from resa.registry import NET
from .resnet import ResNetWrapper 
from .decoder import BUSD, PlainDecoder 


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return (1-d).mean()


class RESA(nn.Module):
    def __init__(self, cfg):
        super(RESA, self).__init__()
        self.iter = cfg.resa.iter
        chan = cfg.resa.input_channel
        fea_stride = cfg.backbone.fea_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = cfg.resa.alpha
        conv_stride = cfg.resa.conv_stride

        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)

            setattr(self, 'conv_r'+str(i), conv_hori1)
            setattr(self, 'conv_l'+str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_d'+str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_u'+str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_r'+str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_l'+str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x



# class ExistHead(nn.Module):
#     def __init__(self, cfg=None):
#         super(ExistHead, self).__init__()
#         self.cfg = cfg

#         self.dropout = nn.Dropout2d(0.1)  # ???
#         self.conv8 = nn.Conv2d(128, cfg.num_classes, 1)

#         stride = cfg.backbone.fea_stride * 2
#         self.fc9 = nn.Linear(
#             int(cfg.num_classes * cfg.img_width / stride * cfg.img_height / stride), 128)
#         self.fc10 = nn.Linear(128, cfg.num_classes-1)

#     def forward(self, x):
#         x = self.dropout(x)
#         x = self.conv8(x)

#         x = F.softmax(x, dim=1)
#         x = F.avg_pool2d(x, 2, stride=2, padding=0)
#         x = x.view(-1, x.numel() // x.shape[0])
#         x = self.fc9(x)
#         x = F.relu(x)
#         x = self.fc10(x)
#         x = torch.sigmoid(x)

#         return x


# @NET.register_module
class RESANet(nn.Module):
    def __init__(self, cfg):
        super(RESANet, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetWrapper(cfg)
        self.resa = RESA(cfg)
        self.decoder = eval(cfg.decoder)(cfg)
        # self.heads = ExistHead(cfg) 

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, target=None):
        fea = self.backbone(img)
        fea = self.resa(fea)
        seg_pred = self.decoder(fea)
        # exist = self.heads(fea)

        # output = {'seg': seg, 'exist': exist}

        # add --> loss
        if seg_gt is not None:
            target = F.one_hot(seg_gt, num_classes=self.cfg.num_classes).permute(0, 3, 1, 2)
            loss = dice_loss(F.softmax(
                seg_pred, dim=1)[:, 1:], target[:, 1:])
        # if seg_gt is not None:
        #     loss = self.ce_loss(seg_pred, seg_gt)
        else:
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, loss
