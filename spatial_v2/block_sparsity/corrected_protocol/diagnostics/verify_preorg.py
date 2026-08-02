import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import spatial_wrapper_cnn as swc
import timm, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# Pre-organization arm: spatial-cost finetune BEFORE the first cut (the phase iter_vit.py skipped),
# then the same 20% magnitude-scored block cut + suite-budget finetune (2ep x 20k).
dev='cuda'; B=16; R=224
t0=time.time(); P=lambda s: print('[%7.1fs] %s'%(time.time()-t0,s), flush=True)

tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr)
idx=np.random.RandomState(0).permutation(len(full))[:20000]
trl=DataLoader(Subset(full,idx),64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)

def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
@torch.no_grad()
def acc(m):
    m.eval(); c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100.*c/t
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
def tiling(net):
    D=[]
    for l in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(l)
        def ids(x,y):
            k=np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1)
            _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go,gi=ids(xo,yo),ids(xi,yi); D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D

torch.manual_seed(0); np.random.seed(0)
mb=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
mb.load_state_dict(torch.load('../vitbase_cifar100_base.pt',map_location='cpu')); mb=mb.to(dev)
net=swc.SpatialCNN(mb,gamma=64.,device=dev,block_size=B).to(dev)
P('swap start'); net.swap(block=256); P('swap done')
lays=regl(mb)

# ---- PRE-ORGANIZATION: 2 epochs dense finetune WITH wiring cost, re-swap each epoch ----
opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
for e in range(2):
    mb.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev)
        loss=F.cross_entropy(net(x),y)+net.get_cost()
        loss.backward(); opt.step(); opt.zero_grad()
    P('pre-org epoch %d done, dense acc %.2f'%(e,acc(mb)))
    P('re-swap'); net.swap(block=256)
D=tiling(net)
# diagnostics: how organized did it get?
with torch.no_grad():
    allw=torch.cat([wmat(l).abs().flatten() for l in lays])
    natsp=100.*(allw<1e-3).float().mean().item()
    zb=tb=0
    for l,(Ro,Co) in zip(lays,D):
        bl2=Ro.t()@(wmat(l)**2)@Co; zb+=int((bl2<1e-6).sum()); tb+=int(bl2.numel())
P('after pre-org: natsp %.1f%%  near-empty blocks %.2f%%  dense acc above'%(natsp,100.*zb/tb))

# ---- same 20%% magnitude block cut as the suite ----
masks=[torch.ones_like(wmat(l)) for l in lays]
alln=[]; binfo=[]
for l,(Ro,Co),msk in zip(lays,D,masks):
    W=wmat(l)*msk; sc=(Ro.t()@(W*W)@Co)
    ab=(Ro.t()@msk@Co)>0; binfo.append((Ro,Co,sc,ab)); alln.append(sc[ab].flatten())
pool=torch.cat(alln).clamp(min=0).sqrt()
totblk=sum(int(ab.numel()) for *_,ab in binfo)
thr=torch.kthvalue(pool,min(int(0.20*totblk),pool.numel())).values.item()
for (Ro,Co,sc,ab),msk in zip(binfo,masks):
    zb2=((sc.sqrt()<=thr)&ab).float(); msk.mul_(1.-(Ro@zb2@Co.t()))
for l,msk in zip(lays,masks):
    with torch.no_grad(): l.weight.mul_(emask(l,msk))
P('PRE-ORG mag post-cut (no ft): %.2f   [blind mag cut was 0.77]'%acc(mb))

# ---- suite-budget finetune (2ep x 20k) ----
opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2*len(trl))
for e in range(2):
    mb.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev)
        loss=F.cross_entropy(net(x),y)+net.get_cost()
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        with torch.no_grad():
            for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
    P('post-cut ft epoch %d done'%e)
zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
P('PRE-ORG FINAL: blk-sp %.1f%%  acc %.2f   [suite spatial@20 was 83.3, taylor@20 was 89.2]'%(100.*zb/tb,acc(mb)))
