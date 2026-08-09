import time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import timm,torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader
dev='cuda'; R=224; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
trl=DataLoader(torchvision.datasets.CIFAR100('./data',train=True,transform=tr),64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)
m=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); m.fc=nn.Linear(512,100); m=m.to(dev)
@torch.no_grad()
def acc():
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
t0=time.time(); opt=torch.optim.AdamW(m.parameters(),1e-4,weight_decay=0.05)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4*len(trl))
for e in range(4):
    m.train()
    for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); sched.step(); opt.zero_grad()
    print('epoch %d acc %.1f (%.0fs)'%(e,acc(),time.time()-t0),flush=True)
torch.save(m.state_dict(),__import__('os').environ.get('SN_OUT','/home/judah/spatial_neurons')+'/rn18_cifar100_base.pt')
print('saved rn18_cifar100_base.pt | final %.1f'%acc(),flush=True)
print('done',flush=True)
