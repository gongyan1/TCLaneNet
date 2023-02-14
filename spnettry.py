import torch
from torchstat import stat
from models.spnet import SPNet
from thop import profile
import time
mod=SPNet(nclass=2,backbone='resnet50',pretrained=None).cuda()
mod.eval()

img=torch.randn([1,3,288,800]).cuda()
time_t=0
for i in range(100):
    a=time.time()
    output=mod(img)
    b=time.time()
    time_t+=(b-a)
print(time_t/100)
'''
stat(mod, (3, 288, 800))'''

