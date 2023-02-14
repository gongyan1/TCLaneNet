import argparse
import json
import os
import shutil
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile
from TCLaneNet import TCLaneNet
from config import *
import dataset
from model_ENET_SAD import ENet_SAD
from torch.utils.tensorboard import SummaryWriter
from utils.transforms import *
from model import SCNN, LaneNet
from resa.resa import RESANet
from shutil import copy
from models.spnet import SPNet
from resa_configs import resa_config,culane,tusimple
from thop import profile
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/SPnet")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
log_writer = SummaryWriter()

exp_dir = args.exp_dir
while exp_dir[-1]=='/':
    exp_dir = exp_dir[:-1]
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

device = torch.device(exp_cfg['device'])

# ------------ test data ------------
# mean, std
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
transform_test_img = Resize(resize_shape)
transform_test_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_test = Compose(transform_test_img, transform_test_x)
test_dataset = Dataset_Type(Dataset_Path[dataset_name], "test", transform_test)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=test_dataset.collate, num_workers=0)
# ------------ preparation ------------
if exp_cfg['model'] == "enet_sad":
    net = ENet_SAD(resize_shape, sad=True)
elif exp_cfg['model'] == "enet":
    net = ENet_SAD(resize_shape, sad=False)
elif exp_cfg['model'] == "TCLaneNet":
    net = TCLaneNet(resize_shape, sad=True)
elif exp_cfg['model'] == "lanenet":
    net = LaneNet()
elif exp_cfg['model'] == "scnn":
    net = SCNN(input_size=[800,288],pretrained=False)
elif exp_cfg['model'] == "resa":
    net = RESANet(culane)
elif exp_cfg['model'] == "SPnet":
    net = SPNet(nclass=2,backbone='resnet50',pretrained=None)
else:
    raise Exception("Model does not match.")
print(os.path.split(exp_cfg['evaluate']['model_path'])[0])
print(exp_cfg['model'])

net=net.cuda()
print(exp_cfg['evaluate']['model_path'])
checkpoint = torch.load(exp_cfg['evaluate']['model_path'], map_location=exp_cfg['device'])
net.load_state_dict(checkpoint['net'])

net = net.to(device)
net = torch.nn.DataParallel(net)

best_val_loss = 1e6

def compute_scores(lane, gt):
    TP, TN, FN, FP = 0.01, 0.01, 0.01, 0.01
    TP += float(np.sum((lane == 1) & (gt == 1)))
    TN += float(np.sum((lane == 0) & (gt == 0)))
    FN += float(np.sum((lane == 0) & (gt == 1)))
    FP += float(np.sum((lane == 1) & (gt == 0)))

    score = dict()
    score['recall'] = TP / (TP + FN)
    score['precision'] = TP / (TP + FP)
    score['F1'] = (2 * score['precision'] * score['recall']) / (score['precision'] + score['recall'])
    score['F2'] = (5 * score['precision'] * score['recall']) / (4 * score['precision'] + score['recall'])
    score['acc'] = (TP + TN) / (TP + TN + FP + FN)
    score['Bacc'] = (TP / (TP + FN) + TN / (TN + FP)) / 2
    score['miou'] = TP / (FP+FN+TP)

    return score


def test(epoch):
    global best_val_loss

    print("Test Epoch: {}".format(epoch))

    net.eval()
    val_loss = 0
    recall_list, prec_list, acc_list, F1_list, miou_list = [],[],[],[],[]
    progressbar = tqdm(range(len(test_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)

            target = sample['target'].to(device)
            if exp_cfg['model'] == "lanenet":
                seg_pred, loss = net(img, segLabel, target=target)  #

                if isinstance(net, torch.nn.DataParallel):
                    loss = loss.sum()

                val_loss += loss.item()

                progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
                progressbar.update(1)
            elif exp_cfg['model'] == "SPnet":
                seg_pred = net(img, segLabel)  #

                #if isinstance(net, torch.nn.DataParallel):
                #    loss = loss.sum()
                #val_loss += loss.item()

                #progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
                #progressbar.update(1)
            else:

                if exp_cfg['model']=='enet_sad' or exp_cfg['model']=='enet':
                    seg_pred, loss = net(img, segLabel)
                else:
                    seg_pred, loss = net(img, segLabel, target=target)

                if isinstance(net, torch.nn.DataParallel):
                    loss = loss.sum()
                
                # val_loss += (loss.item() + lossclass.item())

                progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
                progressbar.update(1)

            seg_pred = seg_pred.detach().cpu().numpy()
            lane = np.argmax(seg_pred, axis=1)
            segLabel = segLabel.detach().cpu().numpy()
            lane[lane > 0] = 1; segLabel[segLabel > 0] = 1
            score = (compute_scores(lane, segLabel))
            recall_list.append(score['recall']); prec_list.append(score['precision'])
            F1_list.append(score['F1']); acc_list.append(score['acc']); miou_list.append([score['miou']])
            lane[lane>0] = 255


            # visualize test images
            gap_num = 1
            if batch_idx%gap_num == 0:
                origin_imgs = []
                seg_pred = seg_pred

                for b in range(len(img)):
                    img_name = sample['img_name'][b]
                    img = cv2.imread(img_name)
                    img = transform_test_img({'img': img})['img']

                    lane_img = np.zeros_like(img)
                    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')

                    coord_mask = np.argmax(seg_pred[b], axis=0)
                    Image.fromarray((coord_mask*255).astype('uint8')).save('result/'+os.path.split(img_name)[1])

                    for i in range(0, 4):
                        lane_img[coord_mask==(i+1)] = color[i]
                    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
                    # cv2.imshow("", cv2.resize(img, (768,432)))#1200, 1920, 3
                    # cv2.waitKey(1)


    progressbar.close()
    log_writer.add_scalar('recall/val', float(np.array(recall_list).mean()), epoch)
    log_writer.add_scalar('prec/val', float(np.array(prec_list).mean()), epoch)
    log_writer.add_scalar('F1/val', float(np.array(F1_list).mean()), epoch)
    log_writer.add_scalar('acc/val', float(np.array(acc_list).mean()), epoch)
    
    print("test: {} \t recall: {:.4} \t prec: {:.4} \t F1: {:.4} \t acc: {:.4} \t miou: {:.4}".format(
            epoch, 
            np.array(recall_list).mean(), 
            np.array(prec_list).mean(), 
            np.array(F1_list).mean(), 
            np.array(acc_list).mean(), 
            np.array(miou_list).mean(), 
        ))

    print("------------------------\n")


def main():
    test(0)


if __name__ == "__main__":
    main()
#test: SPNET 	 recall: 0.9439 	 prec: 0.9717 	 F1: 0.9575 	 acc: 0.9931 	 miou: 0.9187