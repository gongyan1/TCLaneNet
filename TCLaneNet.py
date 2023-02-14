# Enet pytorch code retrieved from https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils.utils import mIoULoss, to_one_hot

# add --> modules
from conditioned_modules import *

# add --> Focus
from focus import Focus

class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.
    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.
    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.
    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.
    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.
    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.
    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class ENet(nn.Module):
    """Generate the ENet model.
    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.
    """

    def __init__(self, num_classes=5, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.scale_background = 0.4

        # Loss scale factor for ENet w/o SAD
        self.scale_seg = 1.0

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()
        self.iou_loss = mIoULoss(n_classes=4)
        self.class_loss = nn.CrossEntropyLoss()

        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        # # add --> FC module
        # self.AvgPool = nn.AvgPool2d(7)
        # self.FC_initial = nn.Sequential(
        #     nn.Linear(4096, 4096), 
        #     nn.ReLU(),
        #     nn.Dropout(0.2), 
        #     nn.Linear(4096, 1024), 
        #     nn.ReLU(), 
        #     nn.Dropout(0.2), 
        # )
        self.AvgPool = nn.AvgPool2d(14)
        self.FC_initial = nn.Sequential(
            nn.Linear(4480, 4096), 
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(4096, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
        )


        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # add --> Conditioning1
        self.gamma1 = nn.Sequential(
            nn.Linear(1024, 64), 
            nn.ReLU(), 
        )
        self.beta1 = nn.Sequential(
            nn.Linear(1024, 64), 
            nn.ReLU(), 
        )
        self.condition1 = nn.Sequential(
            nn.ReLU6(), 
        )


        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # add --> Conditioning2
        self.gamma2 = nn.Sequential(
            nn.Linear(1024, 128), 
            nn.ReLU(), 
        )
        self.beta2 = nn.Sequential(
            nn.Linear(1024, 128), 
            nn.ReLU(), 
        )
        self.condition2 = nn.Sequential(
            nn.ReLU6(), 
        )


        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # add --> Conditioning3
        self.gamma3 = nn.Sequential(
            nn.Linear(1024, 128), 
            nn.ReLU(), 
        )
        self.beta3 = nn.Sequential(
            nn.Linear(1024, 128), 
            nn.ReLU(), 
        )
        self.condition3 = nn.Sequential(
            nn.ReLU6(), 
        )


        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)


        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 2), 
        )


    def forward(self, x, seg_gt=None, target=None):
        # Initial block
        input_size = x.size()
        img = x
        x = self.initial_block(x)

        # add --> FC initial
        FC_input = self.AvgPool(x)
        FC_input = FC_input.view(FC_input.size(0), -1)
        FC_initial = self.FC_initial(FC_input)


        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # add --> Conditioning1
        gamma1 = self.gamma1(FC_initial)
        gamma1 = (gamma1.unsqueeze(2)).unsqueeze(2)
        gamma1 = gamma1.expand(x.shape)
        beta1 = self.beta1(FC_initial)
        beta1 = (beta1.unsqueeze(2)).unsqueeze(2)
        beta1 = beta1.expand(x.shape)
        x = self.condition1(((torch.ones_like(x)-gamma1) * x + beta1))   # torch.ones_like(x) should add cuda

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # add --> Conditioning2
        gamma2 = self.gamma2(FC_initial)
        gamma2 = (gamma2.unsqueeze(2)).unsqueeze(2)
        gamma2 = gamma2.expand(x.shape)
        beta2 = self.beta2(FC_initial)
        beta2 = (beta2.unsqueeze(2)).unsqueeze(2)
        beta2 = beta2.expand(x.shape)
        x = self.condition2(((torch.ones_like(x)-gamma2) * x + beta2))   # torch.ones_like(x) should be in cuda


        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # add --> Conditioning3
        gamma3 = self.gamma3(FC_initial)
        gamma3 = (gamma3.unsqueeze(2)).unsqueeze(2)
        gamma3 = gamma3.expand(x.shape)
        beta3 = self.beta3(FC_initial)
        beta3 = (beta3.unsqueeze(2)).unsqueeze(2)
        beta3 = beta3.expand(x.shape)
        x = self.condition3(((torch.ones_like(x)-gamma3) * x + beta3))   # torch.ones_like(x) should add cuda


        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        seg_pred = self.transposed_conv(x, output_size=input_size)

        # add classifier
        class_out = self.classifier(FC_initial)

        # loss calculation
        if seg_gt is not None:
            # # L = L_seg + a * L_iou + b * L_exist + c * L_distill
            # if self.sad:
            #     loss_seg = self.ce_loss(seg_pred, seg_gt)
            #     seg_gt_onehot = to_one_hot(seg_gt, 5)
            #     loss_iou = self.iou_loss(seg_pred[:, 1:self.num_classes, :, :], seg_gt_onehot[:, 1:self.num_classes, :, :])
            #     loss = loss_seg * self.scale_sad_seg + loss_iou * self.scale_sad_iou
            # else:
            #     loss_seg = self.ce_loss(seg_pred, seg_gt)
            #     loss = loss_seg * self.scale_seg
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss = loss_seg * self.scale_seg

        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)


        # classifier loss
        score = F.softmax(class_out, dim=1)
        class_loss = self.class_loss(score, target)    # !!!

        loss += class_loss

        return seg_pred, loss


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=1, device='cpu'):
        super(SpatialSoftmax, self).__init__()

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature).to(device)
        else:
            self.temperature = 1.

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1, feature.shape[1] * feature.shape[2])
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)

        return softmax_attention

#---------------
class twodConvSum(nn.Module):
    def __init__(self,H,W):
        super(twodConvSum, self).__init__()
        self.convW = nn.Sequential(nn.Conv2d(2, 2,kernel_size=[1,W],stride=1))
        self.convH = nn.Sequential(nn.Conv2d(2, 2, kernel_size=[H, 1], stride=1))
        self.final_conv=nn.Sequential(
            nn.Conv2d(2,1,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        for i in range(2):
            x=self.convW(torch.cat((x,x[:,:,:,:-1]),dim=3))
            x=self.convH(torch.cat((x,x[:,:,:-1,:]),dim=2))
        x=self.final_conv(x)
        return x

class twoConvSum(nn.Module):
    def __init__(self,H,W):
        super(twoConvSum, self).__init__()
        self.convW = nn.Sequential(nn.Conv2d(2, 2,kernel_size=[1,W],stride=1))
        self.convH = nn.Sequential(nn.Conv2d(2, 2, kernel_size=[H, 1], stride=1))
        self.final_conv=nn.Sequential(
            nn.Conv2d(2,1,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        for i in range(2):
            x=self.convW(torch.cat((x,x[:,:,:,:-1]),dim=3))
            x=self.convH(torch.cat((x,x[:,:,:-1,:]),dim=2))
        x=self.final_conv(x)
        return x
#---------------

class TCLaneNet(nn.Module):
    """Generate the ENet model.
    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.
    - sad (bool, optional): When ``True``, SAD is added to model
    . If False, SAD is removed.
    """

    def __init__(self, input_size, pretrained=True, sad=False, encoder_relu=False, decoder_relu=True, weight_share=True):
        super().__init__()
        # Init parameter

        input_w, input_h = input_size
        self.fc_input_feature = 5 * int(input_w / 16) * int(input_h / 16)

        self.num_classes = 5
        self.pretrained = pretrained

        self.scale_background = 0.4

        # Loss scale factor for ENet w/o SAD
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        # Loss scale factor for ENet w SAD
        self.scale_sad_seg = 1.0
        self.scale_sad_iou = 0.1
        self.scale_sad_exist = 0.1
        self.scale_sad_distill = 0.1

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()
        self.iou_loss = mIoULoss(n_classes=4)
        self.class_loss = nn.CrossEntropyLoss()

        # Stage 0 - Initial block
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        self.sad = sad

        # add --> Pooling
        self.AvgPool = nn.AvgPool2d(14)
        self.MaxPool = nn.MaxPool2d(14)

        # add --> Focus
        self.Focus = Focus(c1=32, c2=128)

        # Stage 1 - Encoder (E1)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # add --> Conditioning1
        ''''''
        self.Attention_branch1=nn.Sequential(
            nn.Conv2d(128,16,1),
            nn.ReLU(),
            nn.Conv2d(16,64,1),
            nn.Sigmoid()
        )
        ''''''
        self.Attention_branch2 = nn.Sequential(
            nn.Conv1d(1, 1, 128),
            nn.Sigmoid()
        )
        self.Attention_branch3 = nn.Sequential(
            nn.Conv1d(1, 1, 128),
            nn.Sigmoid()
        )
        ''''''
        self.Attention_branch4 = nn.Sequential(
            nn.Conv1d(1, 1, 128),
            nn.Sigmoid()
        )
        self.SA1 = twodConvSum(72,200)
        self.SA2 = twodConvSum(36, 100)
        self.SA3 = twodConvSum(36, 100)
        self.SA4 = twodConvSum(36, 100)


        # Shared Encoder (E2~E4)
        # Stage 2 - Encoder (E2)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        # add --> Conditioning2
        # Stage 3 - Encoder (E3)
        if weight_share:
            self.regular3_0 = self.regular2_1
            self.dilated3_1 = self.dilated2_2
            self.asymmetric3_2 = self.asymmetric2_3
            self.dilated3_3 = self.dilated2_4
            self.regular3_4 = self.regular2_5
            self.dilated3_5 = self.dilated2_6
            self.asymmetric3_6 = self.asymmetric2_7
            self.dilated3_7 = self.dilated2_8
        else:
            self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
            self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
            self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # add --> Conditioning3

        # Stage 4 - Encoder (E4)
        if weight_share:
            self.regular4_0 = self.regular2_1
            self.dilated4_1 = self.dilated2_2
            self.asymmetric4_2 = self.asymmetric2_3
            self.dilated4_3 = self.dilated2_4
            self.regular4_4 = self.regular2_5
            self.dilated4_5 = self.dilated2_6
            self.asymmetric4_6 = self.asymmetric2_7
            self.dilated4_7 = self.dilated2_8
        else:
            self.regular4_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated4_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric4_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
            self.dilated4_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
            self.regular4_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated4_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric4_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.dilated4_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # add --> Conditioning4


        # Stage 5 - Decoder (D1)
        # self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.upsample4_0 = UpsamplingBottleneck(256, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 6 - Decoder (D2)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(16, self.num_classes, kernel_size=3, stride=2, padding=1, bias=False)

        
        # add --> classifier
        self.classifier1 = nn.Sequential(
            nn.Conv2d(128, 2, 1), 
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(140, 2), 
        )


        # AT_GEN
        if self.sad:
            self.at_gen_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.at_gen_l2_loss = nn.MSELoss(reduction='mean')

        # Lane exist (P1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 5, 1),
            nn.Softmax(dim=1),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def at_gen(self, x1, x2):
        """
        x1 - previous encoder step feature map
        x2 - current encoder step feature map
        """

        # G^2_sum
        sps = SpatialSoftmax(device=x1.device)

        if x1.size() != x2.size():
            x1 = torch.sum(x1 * x1, dim=1)
            x1 = sps(x1)
            x2 = torch.sum(x2 * x2, dim=1, keepdim=True)
            x2 = torch.squeeze(self.at_gen_upsample(x2), dim=1)
            x2 = sps(x2)
        else:
            x1 = torch.sum(x1 * x1, dim=1)
            x1 = sps(x1)
            x2 = torch.sum(x2 * x2, dim=1)
            x2 = sps(x2)

        loss = self.at_gen_l2_loss(x1, x2)
        return loss




    def forward(self, img, seg_gt=None, exist_gt=None, sad_loss=False, target=None):#
        # Stage 0 - Initial block


        input_size = img.size()
        x_0 = self.initial_block(img)


        # add --> Focus init
        Pool1 = self.AvgPool(x_0)

        Pool2 = self.MaxPool(x_0)
        c0 = torch.cat((Pool1, Pool2), dim=1)
        c0 = self.Focus(c0)
        c0_gap = nn.functional.adaptive_avg_pool2d(c0, (1,1))

        # AT-GEN after each E2, E3, E4
        # Stage 1 - Encoder (E1)
        stage1_input_size = x_0.size()
        x, max_indices1_0 = self.downsample1_0(x_0)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x_1 = self.regular1_4(x)

        # add --> Conditioning1
        x_1 = torch.mul(x_1,self.Attention_branch1(c0_gap))
        x_1_sa = self.SA1(x_1)
        x_1 = torch.mul(x_1,x_1_sa)

        c0_gap = c0_gap.squeeze(-1).transpose(-1, -2)
        c0_gap = torch.cat((c0_gap,c0_gap[:,:,:-1]),dim=2)
        # Stage 2 - Encoder (E2)
        stage2_input_size = x_1.size()
        x, max_indices2_0 = self.downsample2_0(x_1)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x_2 = self.dilated2_8(x)
        if self.sad:
            loss_2 = self.at_gen(x_1, x_2)


        # add --> Conditioning2
        x_2 = torch.mul(x_2,self.Attention_branch2(c0_gap).transpose(-1,-2).unsqueeze(-1))
        x_2_sa = self.SA2(x_2)
        x_2 = torch.mul(x_2,x_2_sa)

        # Stage 3 - Encoder (E3)
        x = self.regular3_0(x_2)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x_3 = self.dilated3_7(x)
        if self.sad:
            loss_3 = self.at_gen(x_2, x_3)
        


        # add --> Conditioning3
        x_3 = torch.mul(x_3, self.Attention_branch3(c0_gap).transpose(-1, -2).unsqueeze(-1))
        x_3_sa = self.SA3(x_3)
        x_3 = torch.mul(x_3,x_3_sa)

        # Stage 4 - Encoder (E4)
        x = self.regular3_0(x_3)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x_4 = self.dilated3_7(x)
        if self.sad:
            loss_4 = self.at_gen(x_3, x_4)
        # add --> Conditioning4
        x_4 = torch.mul(x_4, self.Attention_branch4(c0_gap).transpose(-1, -2).unsqueeze(-1))
        x_4_sa = self.SA4(x_4)
        x_4 = torch.mul(x_4,x_4_sa)
        # Concatenate E3, E4
        x_34 = torch.cat((x_3, x_4), dim=1)
        # Stage 4 - Decoder (D1)
        x = self.upsample4_0(x_34, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        # Stage 5 - Decoder (D2)
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        seg_pred = self.transposed_conv(x, output_size=input_size)
        # add classifier
        c0 = self.classifier1(c0)
        c0 = c0.view(c0.size(0), -1)
        class_out = self.classifier2(c0)#

        # loss calculation
        if seg_gt is not None:
            if self.sad:
                loss_seg = self.ce_loss(seg_pred, seg_gt)
                seg_gt_onehot = to_one_hot(seg_gt, 5)
                loss_iou = self.iou_loss(seg_pred[:, 1:self.num_classes, :, :], seg_gt_onehot[:, 1:self.num_classes, :, :])
                loss_distill = loss_2 + loss_3 + loss_4
                loss = loss_seg * self.scale_sad_seg + loss_iou * self.scale_sad_iou
                # Add SAD loss after 40K episodes
                if sad_loss:
                    loss += loss_distill * self.scale_sad_distill

            else:
                loss_seg = self.ce_loss(seg_pred, seg_gt)
                loss = loss_seg * self.scale_seg

        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        # classifier loss
        score = F.softmax(class_out, dim=1)
        class_loss = self.class_loss(score, target)    # !!!
        return seg_pred, loss, class_loss
if __name__ == '__main__':
    tensor = torch.ones((1, 3, 288, 800))
    seg_gt = torch.zeros((1, 288, 800)).long()
    exist_gt = torch.ones((1, 4))
    target = torch.ones(1).long()
    C = {'img': tensor, 'segLabel': seg_gt, 'target': target}
    enet_sad = TCLaneNet((800, 288), sad=True)
    print(enet_sad)
    output, loss,classloss = enet_sad(C, seg_gt=seg_gt, exist_gt=exist_gt, sad_loss=True, target=target)
    total = sum([param.nelement() for param in enet_sad.parameters()])
    print('Prams:', total / 1e6)

