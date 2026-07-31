import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader
LRS=float(sys.argv[1])   # mask-score learning rate
dev='cuda'; B=16; R=160; TGT=0.90; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
trl=DataLoader(torchvision.datasets.CIFAR100('./data',train=True,transform=tr),96,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),512,num_workers=4)
def rn18():
    m=torchvision.models.resnet18(weights='IMAGENET1K_V1'); m.fc=nn.Linear(512,100); return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
def tiling(m):
    D=[]
    for l in regl(m):
        o,inn=wmat(l).shape; D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(inn)//B,int(np.ceil(inn/B)))))
    return D
@torch.no_grad()
def masks_from_scores(scores,D,sp):     # keep top (1-sp) blocks by score -> per-weight masks
    allS=torch.cat([S.flatten() for S in scores]); thr=torch.quantile(allS,sp)
    return [ (Ro@(S>=thr).float()@Co.t()) for S,(Ro,Co) in zip(scores,D) ], thr
@torch.no_grad()
def evala(m,lays,MW):
    W0=[l.weight.detach().clone() for l in lays]
    for l,Mw in zip(lays,MW): l.weight.mul_(emask(l,Mw))
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    for l,w in zip(lays,W0): l.weight.copy_(w)
    return 100.*c/t
t0=time.time()
m=rn18(); opt=torch.optim.Adam(m.parameters(),1e-4)
for e in range(4):
    m.train()
    for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); opt.zero_grad()
lays=regl(m); D=tiling(m)
# init block scores = block L2 magnitude
scores=[torch.sqrt((Ro.t()@(wmat(l)**2)@Co)+1e-12).detach().clone() for l,(Ro,Co) in zip(lays,D)]
print('base %.1f (%.0fs)'%(evala(m,lays,[torch.ones_like(wmat(l)) for l in lays]),time.time()-t0),flush=True)
opt=torch.optim.Adam(m.parameters(),1e-4); res=[]; nb=16
for e in range(nb):
    sp=TGT*(1-(1-e/(nb-1))**3)                       # cubic sparsity schedule 0 -> 0.90
    m.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev)
        MW,_=masks_from_scores(scores,D,sp); W0=[l.weight.detach().clone() for l in lays]
        for l,Mw in zip(lays,MW): l.weight.data.mul_(emask(l,Mw))  # forward through masked weights
        F.cross_entropy(m(x),y).backward()
        with torch.no_grad():
            for l,w in zip(lays,W0): l.weight.copy_(w)             # restore underlying weights
            for l,S,(Ro,Co),w in zip(lays,scores,D,W0):            # MOVEMENT score update (STE)
                g=(l.weight.grad.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight.grad)
                S.add_(-LRS*(Ro.t()@(g*w if not isinstance(l,nn.Conv2d) else g*w.view(l.out_channels,-1))@Co))
        opt.step(); opt.zero_grad()
    MW,_=masks_from_scores(scores,D,sp); a=evala(m,lays,MW)
    zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))  # not used; report scheduled sp
    res.append((100.*sp,a)); print('[movement lrs=%g] ep%2d | blk-sp %4.1f%% | acc %.1f (%.0fs)'%(LRS,e,100*sp,a,time.time()-t0),flush=True)
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/movefaithful_%g.pkl'%LRS,'wb'))
print('done',flush=True)
