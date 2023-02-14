import argparse
import json
import os
import shutil
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from resa_configs.resa_config import Config
from config import *
import dataset
from model import SCNN, LaneNet
from model_ENET_SAD import ENet_SAD
from TCLaneNet import TCLaneNet
from resa.resa import RESANet
from torch.utils.tensorboard import SummaryWriter
from utils.transforms import *
from utils.lr_scheduler import PolyLR
from models.spnet import SPNet
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/SPnet")
    parser.add_argument("--resume", "-r", action="store_true")
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
# ------------ train data ------------
# mean, std
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)
transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(),
                          Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
train_dataset = Dataset_Type(Dataset_Path[dataset_name], "train", transform_train)
print(exp_cfg['dataset']['batch_size'])
train_loader = DataLoader(train_dataset, batch_size=exp_cfg['dataset']['batch_size'], shuffle=True, collate_fn=train_dataset.collate, num_workers=0)
print(len(train_loader))
# ------------ val data ------------
transform_val_img = Resize(resize_shape)
transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_val = Compose(transform_val_img, transform_val_x)
val_dataset = Dataset_Type(Dataset_Path[dataset_name], "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=exp_cfg['dataset']['batch_size'], collate_fn=val_dataset.collate, num_workers=0)

# ------------ preparation ------------
if exp_cfg['model'] == "scnn":
    net = SCNN(resize_shape, pretrained=True)
elif exp_cfg['model'] == "enet_sad":
    net = ENet_SAD(resize_shape, sad=True)
elif exp_cfg['model'] == "TCLaneNet":
    net = TCLaneNet(resize_shape, sad=True)
elif exp_cfg['model'] == "enet":
    net = ENet_SAD(resize_shape, sad=False)
elif exp_cfg['model'] == "resa":
    net = RESANet(cfg=Config.fromfile("resa_configs/mydata.py"))
elif exp_cfg['model'] == "lanenet":
    net = LaneNet()
elif exp_cfg['model'] == 'SPnet':
    net = SPNet(nclass=2,backbone='resnet50',pretrained=None)
else:
    raise Exception("Model not match.")

if exp_cfg['evaluate']['test'] == 'yes':
    checkpoint = torch.load(exp_cfg['evaluate']['model_path'], map_location=exp_cfg['device'])
    net.load_state_dict(checkpoint['net'])

net = net.to(device)
print(net)

optimizer = optim.SGD(net.parameters(), **exp_cfg['optim'])
lr_scheduler = PolyLR(optimizer, 0.9, **exp_cfg['lr_scheduler'])
best_val_loss = 1e6
best_val_iou = 0

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


def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()
    train_loss = 0
    train_loss_class=0
    progressbar = tqdm(range(len(train_loader)))
    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].to(device)
        segLabel = sample['segLabel'].to(device)
        target = sample['target'].to(device)
        exist = None

        optimizer.zero_grad()
        if exp_cfg['model'] == "scnn":
            seg_pred, loss = net(img, segLabel, exist)
        elif exp_cfg['model'] == "enet_sad":
            if (epoch * len(train_loader) + batch_idx) < exp_cfg['sad_start_iter']:
                seg_pred, loss = net(img, segLabel)
            else:
                #print("sad activated")
                seg_pred, loss = net(img, segLabel, exist, True, target)
        elif exp_cfg['model'] == 'enet':
            seg_pred, loss = net(img, segLabel, target)
        elif exp_cfg['model'] == 'resa':
            seg_pred, loss = net(img, segLabel, target)
        elif exp_cfg['model'] == 'lanenet':
            seg_pred, loss = net(img, segLabel)
        elif exp_cfg['model'] == 'TCLaneNet':
            if (epoch * len(train_loader) + batch_idx) < exp_cfg['sad_start_iter']:
                seg_pred, loss, class_loss = net(img, segLabel, exist, False, target)
            else:
                #print("sad activated")
                seg_pred, loss, class_loss = net(img, segLabel, exist, True, target)
        elif exp_cfg['model'] == 'SPnet':
            seg_pred, loss ,auxloss= net(img, segLabel)
        if exp_cfg['model']=='TCLaneNet':
            (loss+class_loss).backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss+=loss.item()
            train_loss_class+=class_loss.item()
            iter_idx = epoch * len(train_loader) + batch_idx
            #train_loss = loss.item()
            progressbar.set_description("batch loss: {:.3f}".format((loss+class_loss).item()))
            progressbar.update(1)

            lr = optimizer.param_groups[0]['lr']
            log_writer.add_scalar('Loss/train', float(loss.item()+class_loss.item()), iter_idx)
        elif exp_cfg['model']=='SPnet':
            (torch.mean(loss)+0.4*torch.mean(auxloss)).backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()

            iter_idx = epoch * len(train_loader) + batch_idx
            # train_loss = loss.item()
            progressbar.set_description("batch loss: {:.3f}".format((loss).item()))
            progressbar.update(1)

            lr = optimizer.param_groups[0]['lr']
            log_writer.add_scalar('Loss/train', float(loss.item() ), iter_idx)
        else:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()
            iter_idx = epoch * len(train_loader) + batch_idx
            # train_loss = loss.item()
            progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)
            lr = optimizer.param_groups[0]['lr']
            log_writer.add_scalar('Loss/train', float(loss.item()), iter_idx)
    progressbar.close()

    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_iou": best_val_iou
        }
        save_name = os.path.join(exp_dir, exp_name + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))
        print(best_val_iou)

    print("------------------------\n")
    if exp_cfg['model']=='TCLaneNet':
        return train_loss / len(train_loader), train_loss_class / len(train_loader)
    else:
        return train_loss / len(train_loader)



def val(epoch):
    global best_val_loss
    global best_val_iou
    print("Val Epoch: {}".format(epoch))
    net.eval()
    val_loss = 0
    val_iou = 0
    recall_list = []; prec_list = []
    acc_list = []; F1_list = []; miou_list = []
    progressbar = tqdm(range(len(val_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)
            target = sample['target'].to(device)
            if exp_cfg['model']=='TCLaneNet':
                seg_pred, loss, class_loss = net(img, segLabel, target=target)
                val_loss += (loss + class_loss).item()
                progressbar.set_description("batch loss: {:.3f}".format((loss + class_loss).item()))
            elif exp_cfg['model']=='SPnet':
                seg_pred = net(img, segLabel)
            else:
                if exp_cfg['model']=='enet_sad' or exp_cfg['model']=='enet':
                    seg_pred, loss = net(img, segLabel)
                else:
                    seg_pred, loss = net(img, segLabel, target=target)
                val_loss += loss.item()
                progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)
            seg_pred = seg_pred.detach().cpu().numpy()
            lane = np.argmax(seg_pred, axis=1)
            segLabel = segLabel.detach().cpu().numpy()
            lane[lane > 0] = 1; segLabel[segLabel > 0] = 1
            score = (compute_scores(lane, segLabel))
            recall_list.append(score['recall']); prec_list.append(score['precision'])
            F1_list.append(score['F1']); acc_list.append(score['acc']); miou_list.append([score['miou']])
            val_iou+=score['miou']

    progressbar.close()
    log_writer.add_scalar('recall/val', float(np.array(recall_list).mean()), epoch)
    log_writer.add_scalar('prec/val', float(np.array(prec_list).mean()), epoch)
    log_writer.add_scalar('F1/val', float(np.array(F1_list).mean()), epoch)
    log_writer.add_scalar('acc/val', float(np.array(acc_list).mean()), epoch)
    print("val: {} \t recall: {:.4} \t prec: {:.4} \t F1: {:.4} \t acc: {:.4} \t miou: {:.4}".format(
            epoch, 
            np.array(recall_list).mean(), 
            np.array(prec_list).mean(), 
            np.array(F1_list).mean(), 
            np.array(acc_list).mean(), 
            np.array(miou_list).mean(), 
        ))
    if exp_cfg['evaluate']['test'] != 'yes':
        print("------------------------\n")
        if val_iou/len(val_loader) > best_val_iou:
            best_val_iou = val_iou/len(val_loader)
            save_name = os.path.join(exp_dir, exp_name + '.pth')
            copy_name = os.path.join(exp_dir, exp_name + '_iou_best.pth')
            shutil.copyfile(save_name, copy_name)

        if exp_cfg['model'] != 'SPnet' and val_loss/len(val_loader) < best_val_loss:
            best_val_loss = val_loss/len(val_loader)
            save_name = os.path.join(exp_dir, exp_name + '.pth')
            copy_name = os.path.join(exp_dir, exp_name + '_loss_best.pth')
            shutil.copyfile(save_name, copy_name)


def main():
    train_epoch_num = 0
    train_epoch_segloss = []
    train_epoch_classloss = []
    if exp_cfg['evaluate']['test'] == 'yes':
        val(0)
        exit()

    global best_val_loss
    global best_val_iou
    if args.resume:
        save_dict = torch.load(os.path.join(exp_dir, exp_name + '.pth'), map_location=exp_cfg['device'])
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict.get("best_val_loss", 1e6)
    else:
        start_epoch = 0

    if exp_cfg['MAX_EPOCHES'] == 'no':
        exp_cfg['MAX_EPOCHES'] = int(np.ceil(exp_cfg['lr_scheduler']['max_iter'] / len(train_loader)))
    
    for epoch in range(start_epoch, exp_cfg['MAX_EPOCHES']):
        train_epoch_num+=1
        if exp_cfg['model'] == 'TCLaneNet':
            train_loss, train_loss_class = train(epoch)
            train_epoch_segloss.append(str(format(train_loss, '.8f')))
            train_epoch_classloss.append(str(format(train_loss_class, '.8f')))
            print('train_loss_seg:', train_loss, 'train_loss_class:', train_loss_class)
        else:
            train_loss= train(epoch)
            train_epoch_segloss.append(str(format(train_loss, '.8f')))
            print('train_loss:', train_loss)

        if epoch % 5 == 0 and epoch <= 140:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)
        elif epoch % 1 ==0 and epoch >140:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)
if __name__ == "__main__":
    main()
