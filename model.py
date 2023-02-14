import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SCNN(nn.Module):
    def __init__(
            self,
            input_size,
            ms_ks=9,
            pretrained=True
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None, target=None):#
        '''
        seg_gt = img['segLabel']
        target = img['target']
        img = img['img']'''
        x = self.backbone(img)
        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer2(x)

        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        x = self.layer3(x)
        x = x.view(-1, self.fc_input_feature)
        # exist_pred = self.fc(x)

        # if seg_gt is not None and exist_gt is not None:
        #     loss_seg = self.ce_loss(seg_pred, seg_gt)
        #     loss_exist = self.bce_loss(exist_pred, exist_gt)
        #     loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        # else:
        #     loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
        #     loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
        #     loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        if seg_gt is not None:
            loss = self.ce_loss(seg_pred, seg_gt)
            exist_pred = 0
            loss_exist = 0
        else:
            # loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, loss

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        input_w, input_h = input_size
        self.fc_input_feature = 5 * int(input_w/16) * int(input_h/16)
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()




class LaneNet(nn.Module):
    def __init__(
            self,
            embed_dim=4,
            delta_v=0.5,
            delta_d=3.0,
            scale_lane_line=1.0,
            scale_var=1.0,
            scale_dist=1.0,
            pretrained=False,
            **kwargs
    ):
        super(LaneNet, self).__init__()
        self.pretrained = pretrained
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

        self.net_init()

        self.scale_seg = scale_lane_line
        self.scale_var = scale_var
        self.scale_dist = scale_dist
        self.scale_reg = 0
        self.seg_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1.]))

    def net_init(self):
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # ----------------- additional conv -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

        # ----------------- embedding -----------------
        self.embedding = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, self.embed_dim, 1)
        )

        # ----------------- binary segmentation -----------------
        self.binary_seg = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, 1)
        )


    def forward(self, img, segLabel=None, target=None):#
        '''
        segLabel=img['segLabel']
        target=img['target']
        img=img['img']'''
        input_size = img.size()

        x = self.backbone(img)
        x = self.layer1(x)

        embedding = self.embedding(x)
        binary_seg = self.binary_seg(x)

        if segLabel is not None:
            var_loss, dist_loss, reg_loss = self.discriminative_loss(embedding, segLabel)
            seg_loss = self.seg_loss(binary_seg, torch.gt(segLabel, 0).type(torch.long))
        else:
            var_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
            dist_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
            seg_loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        loss = seg_loss * self.scale_seg + var_loss * self.scale_var + dist_loss * self.scale_dist

        output = {
            "embedding": embedding,
            "binary_seg": binary_seg,
            "seg_loss": seg_loss,
            "var_loss": var_loss,
            "dist_loss": dist_loss,
            "reg_loss": reg_loss, 
            "loss": loss
        }

        return output["binary_seg"], output["loss"]

    def discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b] # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels!=0]
            num_lanes = len(labels)
            if num_lanes==0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean( F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim,1), dim=0) - self.delta_v)**2 ) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size
        return var_loss, dist_loss, reg_loss