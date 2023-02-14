net = dict(
    type='RESANet',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet50',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    fea_stride=8,
)

resa = dict(
    type='RESA',
    alpha=2.0,
    iter=4,
    input_channel=128,
    conv_stride=9,
)

decoder = 'PlainDecoder'        

trainer = dict(
    type='RESA'
)

evaluator = dict(
    type='CULane',        
)

optimizer = dict(
  type='sgd',
  lr=0.025,
  weight_decay=1e-4,
  momentum=0.9
)

epochs = 12
batch_size = 2
total_iter = (88880 // batch_size) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

loss_type = 'dice_loss'
seg_loss_weight = 2.
eval_ep = 6
save_ep = epochs

bg_weight = 0.4

img_norm = dict(
    mean=[0.3598, 0.3653, 0.3662],
    std=[0.2573, 0.2663, 0.2756]
)

img_height = 288
img_width = 800
cut_height = 240 

dataset_path = r'E:\LaneDetection\dataset\vibration2000\list'
dataset = dict(
    train=dict(
        type='MyData',
        img_path=dataset_path,
        data_list='train_gt.txt',
    ),
    val=dict(
        type='MyData',
        img_path=dataset_path,
        data_list='val_gt.txt',
    ),
    test=dict(
        type='MyData',
        img_path=dataset_path,
        data_list='test_gt.txt',
    )
)


workers = 16
num_classes = 4 + 1
ignore_label = 255
log_interval = 500
