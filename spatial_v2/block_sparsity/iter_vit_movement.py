import sys,copy,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import timm,torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader,Subset
KD=(len(sys.argv)>1 and sys.argv[1]=='kd'); LRS=0.01
dev='cuda'; B=16; R=224; TGT=0.90; torch.manual_seed(0); np.random.seed(0)
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
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
def tiling(m):
    D=[]
    for l in regl(m):
        o,i=wmat(l).shape; D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(i)//B,int(np.ceil(i/B)))))
    return D
@torch.no_grad()
def masks_from_scores(scores,D,sp):
    allS=torch.cat([S.flatten() for S in scores]); thr=torch.quantile(allS,sp)
    return [(Ro@(S>=thr).float()@Co.t()) for S,(Ro,Co) in zip(scores,D)]
@torch.no_grad()
def evala(m,lays,MW):
    W0=[l.weight.detach().clone() for l in lays]
    for l,Mw in zip(lays,MW): l.weight.mul_(emask(l,Mw))
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    for l,w in zip(lays,W0): l.weight.copy_(w)
    return 100.*c/t
t0=time.time(); m=vit(); lays=regl(m); D=tiling(m)
teacher=copy.deepcopy(m); teacher.eval()
for p in teacher.parameters(): p.requires_grad=False
scores=[torch.sqrt((Ro.t()@(wmat(l)**2)@Co)+1e-12).detach().clone() for l,(Ro,Co) in zip(lays,D)]
print('base %.1f (%.0fs)'%(evala(m,lays,[torch.ones_like(wmat(l)) for l in lays]),time.time()-t0),flush=True)
opt=torch.optim.AdamW(m.parameters(),5e-5,weight_decay=0.05); res=[]; nb=16
for e in range(nb):
    sp=TGT*(1-(1-e/(nb-1))**3); m.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev); MW=masks_from_scores(scores,D,sp); W0=[l.weight.detach().clone() for l in lays]
        for l,Mw in zip(lays,MW): l.weight.data.mul_(emask(l,Mw))
        st=m(x)
        if KD:
            with torch.no_grad(): tea=teacher(x)
            loss=F.cross_entropy(st,y)+0.5*F.kl_div(F.log_softmax(st/2,1),F.softmax(tea/2,1),reduction='batchmean')*4
        else: loss=F.cross_entropy(st,y)
        loss.backward()
        with torch.no_grad():
            for l,w in zip(lays,W0): l.weight.copy_(w)
            for l,S,(Ro,Co),w in zip(lays,scores,D,W0):
                g=(l.weight.grad.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight.grad)
                S.add_(-LRS*(Ro.t()@(g*w if not isinstance(l,nn.Conv2d) else g*w.view(l.out_channels,-1))@Co))
        opt.step(); opt.zero_grad()
    MW=masks_from_scores(scores,D,sp); a=evala(m,lays,MW)
    res.append((100.*sp,a)); print('[vit-movement%s] ep%2d | blk-sp %4.1f%% | acc %.1f (%.0fs)'%('+KD' if KD else '',e,100*sp,a,time.time()-t0),flush=True)
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/vit_movement_%s.pkl'%('kd' if KD else 'plain'),'wb'))
print('done',flush=True)
