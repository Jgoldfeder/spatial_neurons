import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader
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
def exempt_mask(lays):   # per-layer allocation: protect first conv, 1x1 downsample convs, and fc
    ex=[]
    for i,l in enumerate(lays):
        e = (i==0) or isinstance(l,nn.Linear) or (isinstance(l,nn.Conv2d) and l.kernel_size==(1,1))
        ex.append(e)
    return ex
def iterative(m,D,levels,F_ep,tag):
    lays=regl(m); EX=exempt_mask(lays); masks=[torch.ones_like(wmat(l)) for l in lays]; out=[]
    for target in levels:
        alln=[]; binfo=[]
        for l,(Ro,Co),msk,ex in zip(lays,D,masks,EX):
            W=wmat(l)*msk; b=(Ro.t()@(W*W)@Co); ab=(Ro.t()@msk@Co)>0
            binfo.append((Ro,Co,b,ab,ex))
            if not ex: alln.append(b[ab].flatten())      # only PRUNABLE layers enter the ranking
        pool=torch.cat(alln).clamp(min=0).sqrt()
        totblk=sum(int(ab.numel()) for *_,ab,ex in binfo)                     # count over ALL blocks
        already=sum(int((~ab).sum()) for *_,ab,ex in binfo)
        need=max(0,int(target/100*totblk)-already)
        if need>0 and pool.numel()>0:
            thr=torch.kthvalue(pool,min(need,pool.numel())).values.item()
            for (Ro,Co,b,ab,ex),msk in zip(binfo,masks):
                if ex: continue
                zb=((b.sqrt()<=thr)&ab).float(); msk.mul_(1.-(Ro@zb@Co.t()))
        for l,msk in zip(lays,masks):
            with torch.no_grad(): l.weight.mul_(emask(l,msk))
        opt=torch.optim.Adam(m.parameters(),5e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,F_ep*len(trl))
        for e in range(F_ep):
            m.train()
            for x,y in trl:
                x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); sched.step(); opt.zero_grad()
                with torch.no_grad():
                    for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
        zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
        tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
        a=acc(m); out.append((100.*zb/tb,a)); print('[mag-perlayer] tgt %2d%% | blk-sp %4.1f%% | acc %.1f'%(target,100.*zb/tb,a),flush=True)
    return out
LEV=[20,40,55,65,75,80,85,90]; t0=time.time()
mb=rn18(); opt=torch.optim.Adam(mb.parameters(),1e-4)
for e in range(4):
    mb.train()
    for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(mb(x),y).backward(); opt.step(); opt.zero_grad()
print('base %.1f (exempt %d/%d layers) (%.0fs)'%(acc(mb),sum(exempt_mask(regl(mb))),len(regl(mb)),time.time()-t0),flush=True)
res=iterative(mb,tiling(mb),LEV,2,'mag-perlayer')
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/mag_perlayer.pkl','wb'))
print('done',flush=True)
