import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import timm,torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader,Subset
SP=float(sys.argv[1]); dev='cuda'; B=16; R=224; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr); idx=np.random.RandomState(0).permutation(len(full))[:20000]
trl=DataLoader(Subset(full,idx),64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)
def vit():
    m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
    m.load_state_dict(torch.load('/home/judah/spatial_v2/block_sparsity/vitbase_cifar100_base.pt',map_location='cpu')); return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def gmat(l): return l.weight.grad.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight.grad
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
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
        o,i=wmat(l).shape; D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(i)//B,int(np.ceil(i/B)))))
    return D
t0=time.time(); m=vit(); print('base %.1f (%.0fs)'%(acc(m),time.time()-t0),flush=True)
lays=regl(m); D=tiling(m)
@torch.no_grad()
def block_mag(): return [torch.sqrt((Ro.t()@(wmat(l)**2)@Co)+1e-12) for l,(Ro,Co) in zip(lays,D)]
bm=block_mag(); allm=torch.cat([b.flatten() for b in bm]); thr=torch.quantile(allm,SP)
active=[(b>thr) for b in bm]
@torch.no_grad()
def enforce():
    for l,(Ro,Co),ac in zip(lays,D,active): l.weight.mul_(emask(l,(Ro@ac.float()@Co.t())))
enforce()
opt=torch.optim.AdamW(m.parameters(),5e-5,weight_decay=0.05); res=[]; nb=16
for e in range(nb):
    f=0.3*0.5*(1+np.cos(np.pi*e/nb))
    if e>0 and f>0.01:
        m.zero_grad()
        for i,(x,y) in enumerate(trl):
            F.cross_entropy(m(x.to(dev)),y.to(dev)).backward()
            if i>=10: break
        with torch.no_grad():
            bg=[torch.sqrt((Ro.t()@(gmat(l)**2)@Co)+1e-12) for l,(Ro,Co) in zip(lays,D)]; bmg=block_mag()
            nact=int(sum(int(a.sum()) for a in active)); k=int(f*nact)
            act_mags=torch.cat([bmg[i][active[i]] for i in range(len(lays))])
            if k>0 and act_mags.numel()>k:
                dthr=torch.kthvalue(act_mags,k).values.item()
                inact_g=torch.cat([bg[i][~active[i]] for i in range(len(lays))])
                gthr=torch.kthvalue(inact_g,max(1,inact_g.numel()-k+1)).values.item()
                for i in range(len(lays)):
                    drop=(bmg[i]<=dthr)&active[i]; grow=(bg[i]>=gthr)&(~active[i]); active[i]=(active[i]&~drop)|grow
        m.zero_grad(); enforce()
    m.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); opt.zero_grad(); enforce()
    zb=sum(int((~a).sum()) for a in active); tb=sum(int(a.numel()) for a in active)
    a=acc(m); res.append((100.*zb/tb,a)); print('[vit-RigL sp=%.0f] ep%2d | blk-sp %4.1f%% | acc %.1f (%.0fs)'%(SP*100,e,100.*zb/tb,a,time.time()-t0),flush=True)
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/vit_rigl_%g.pkl'%SP,'wb'))
print('done',flush=True)
