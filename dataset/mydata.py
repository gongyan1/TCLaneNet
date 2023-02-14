import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset


class mydata(Dataset):
    
    def __init__(self, path, image_set, transforms=None):
        super(mydata, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.createIndex()


    def createIndex(self):
        # gt_adverse_light        gt_yes
        listfile = os.path.join(self.data_dir_path, "list", "{}_gt.txt".format(self.image_set))
        self.img_list = []
        self.segLabel_list = []
        self.target_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0]))   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(os.path.join(self.data_dir_path, l[1]))
                self.target_list.append(torch.Tensor([int(l[2])]).long())

    def __getitem__(self, idx):
        if self.img_list[idx].split('/')[-1].split('.png')[0] != self.segLabel_list[idx].split('/')[-1].split('.mask.png')[0]:
            print('error: the label does not match the image')
            print(self.img_list[idx], '\n', self.segLabel_list[idx])
            exit()
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segLabel = cv2.imread(self.segLabel_list[idx])
        segLabel = cv2.cvtColor(segLabel, cv2.COLOR_RGB2GRAY)
        segLabel[segLabel>=255] = 255; segLabel[segLabel<255] = 0
        segLabel = cv2.cvtColor(segLabel, cv2.COLOR_GRAY2RGB)[:,:,0]
        segLabel[segLabel>=255] = 1; segLabel[segLabel<1] = 0

        sample = {'img': img,
                  'segLabel': segLabel,
                  'target': self.target_list[idx], 
                  'img_name': self.img_list[idx]}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
            target = torch.cat([b['target'] for b in batch])
        else:
            img = [b['img'] for b in batch]
            target = [b['target'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]

        samples = {'img': img,
                  'segLabel': segLabel,
                  'target': target, 
                  'img_name': [x['img_name'] for x in batch]}

        return samples