import sys,time,copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import torchvision,torchvision.transforms as T,torch_pruning as tp
from torch.utils.data import DataLoader
METHOD=sys.argv[1]  # mag | taylor | fpgm
dev='cuda'; R=160; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
trl=DataLoader(torchvision.datasets.CIFAR100('./data',train=True,transform=tr),96,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),512,num_workers=4)
def rn18():
    m=torchvision.models.resnet18(weights='IMAGENET1K_V1'); m.fc=nn.Linear(512,100); return m.to(dev)
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
def finetune(m,ep,lr):
    opt=torch.optim.Adam(m.parameters(),lr); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep*len(trl))
    for e in range(ep):
        m.train()
        for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); sched.step(); opt.zero_grad()
IMP={'mag':lambda:tp.importance.GroupMagnitudeImportance(p=2),
     'taylor':lambda:tp.importance.GroupTaylorImportance(),
     'fpgm':lambda:tp.importance.FPGMImportance(p=2)}[METHOD]
t0=time.time()
m=rn18(); finetune(m,4,1e-4); pbase=sum(p.numel() for p in m.parameters())   # 4-epoch base (same as block arms)
print('base %.1f | %.2fM params (%.0fs)'%(acc(m),pbase/1e6,time.time()-t0),flush=True)
ex=torch.randn(1,3,R,R).to(dev)
# GRADUAL channel pruning: 8 steps to ratio 0.75, 2 finetune epochs between each = SAME protocol as block arms
pruner=tp.pruner.MetaPruner(m,ex,importance=IMP(),pruning_ratio=0.75,iterative_steps=8,ignored_layers=[x for x in m.modules() if isinstance(x,nn.Linear)])
res=[]
for step in range(8):
    if METHOD=='taylor':                       # Taylor needs grads accumulated before pruning
        m.zero_grad()
        for i,(x,y) in enumerate(trl):
            F.cross_entropy(m(x.to(dev)),y.to(dev)).backward()
            if i>=15: break
    pruner.step()
    finetune(m,4,5e-4)
    p=sum(pp.numel() for pp in m.parameters()); a=acc(m)
    res.append((100.*p/pbase,a)); print('[chanGEN-%s] step %d | params %.1f%% dense | acc %.1f (%.0fs)'%(METHOD,step+1,100.*p/pbase,a,time.time()-t0),flush=True)
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/chan_gen_%s.pkl'%METHOD,'wb'))
print('done',flush=True)
