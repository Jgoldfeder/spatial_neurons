import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader
TAU=float(sys.argv[1])   # per-step group soft-threshold strength
dev='cuda'; B=16; R=160; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
trl=DataLoader(torchvision.datasets.CIFAR100('./data',train=True,transform=tr),96,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),512,num_workers=4)
def rn18():
    m=torchvision.models.resnet18(weights='IMAGENET1K_V1'); m.fc=nn.Linear(512,100); return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def emask(l,msk): return msk.view_as(l.weight) if isinstance(l,nn.Conv2d) else msk
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
def tiling(m):
    D=[]
    for l in regl(m):
        o,inn=wmat(l).shape; D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(inn)//B,int(np.ceil(inn/B)))))
    return D
@torch.no_grad()
def prox_group_soft_threshold(lays,D,tau):
    # W_block <- max(0, 1 - tau/||W_block||)*W_block  (drives small blocks to EXACT zero)
    for l,(Ro,Co) in zip(lays,D):
        W=wmat(l); bn=torch.sqrt((Ro.t()@(W*W)@Co)+1e-12)
        shrink=torch.clamp(1.-tau/bn,min=0.)
        W.mul_(Ro@shrink@Co.t())
@torch.no_grad()
def block_sparsity(lays,D):
    zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
    tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D); return 100.*zb/tb
t0=time.time()
m=rn18(); opt=torch.optim.Adam(m.parameters(),1e-4)
for e in range(4):
    m.train()
    for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); opt.zero_grad()
print('base %.1f (%.0fs)'%(acc(m),time.time()-t0),flush=True)
lays=regl(m); D=tiling(m); res=[]
opt=torch.optim.Adam(m.parameters(),1e-4)
for e in range(16):    # 16 finetune epochs with proximal group-lasso each step (matches block-arm budget)
    m.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); opt.zero_grad()
        prox_group_soft_threshold(lays,D,TAU)
    bs=block_sparsity(lays,D); a=acc(m); res.append((bs,a))
    print('[prox-glasso tau=%g] ep%2d | blk-sp %4.1f%% | acc %.1f (%.0fs)'%(TAU,e,bs,a,time.time()-t0),flush=True)
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/proxglasso_%g.pkl'%TAU,'wb'))
print('done',flush=True)
