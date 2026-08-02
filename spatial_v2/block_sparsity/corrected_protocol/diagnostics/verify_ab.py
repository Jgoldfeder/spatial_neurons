import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import spatial_wrapper_cnn as swc
import timm, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

dev='cuda'; B=16; R=224
t0=time.time(); P=lambda s: print('[%7.1fs] %s'%(time.time()-t0,s), flush=True)

tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr)
sub_idx=np.random.RandomState(0).permutation(len(full))[:20000]
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)

def vit():
    m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
    m.load_state_dict(torch.load('../vitbase_cifar100_base.pt',map_location='cpu')); return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def gmat(l): return l.weight.grad.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight.grad
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

# ---------- (A) swap function-preservation ----------
torch.manual_seed(0); np.random.seed(0)
mb=vit(); a0=acc(mb); P('A: dense base acc              %.2f'%a0)
net=swc.SpatialCNN(mb,gamma=64.,device=dev,block_size=B).to(dev)
a1=acc(mb); P('A: after wrap (no swap)        %.2f'%a1)
net.swap(block=256)
a2=acc(mb); a3=acc(net); P('A: after swap  model=%.2f  net-forward=%.2f'%(a2,a3))
P('A VERDICT: %s'%('function preserved' if abs(a2-a0)<1e-9 else 'CHANGED by %.2f'%(a2-a0)))
del mb, net; torch.cuda.empty_cache()

# ---------- (B) 20%-target reruns of the spatial arm ----------
def run(tag, scoring, ft_epochs, ft_full):
    torch.manual_seed(0); np.random.seed(0)
    mb=vit(); net=swc.SpatialCNN(mb,gamma=64.,device=dev,block_size=B).to(dev); net.swap(block=256)
    D=tiling(net); lays=regl(mb); masks=[torch.ones_like(wmat(l)) for l in lays]
    idx = np.arange(len(full)) if ft_full else sub_idx
    trl=DataLoader(Subset(full,idx),64,shuffle=True,num_workers=8,pin_memory=True)
    if scoring=='taylor':
        mb.zero_grad()
        for i,(x,y) in enumerate(trl):
            F.cross_entropy(net(x.to(dev)),y.to(dev)).backward()
            if i>=10: break
    alln=[]; binfo=[]
    for l,(Ro,Co),msk in zip(lays,D,masks):
        W=wmat(l)*msk
        sc=(Ro.t()@((gmat(l)*W)**2)@Co) if scoring=='taylor' else (Ro.t()@(W*W)@Co)
        ab=(Ro.t()@msk@Co)>0; binfo.append((Ro,Co,sc,ab)); alln.append(sc[ab].flatten())
    mb.zero_grad()
    pool=torch.cat(alln).clamp(min=0).sqrt()
    totblk=sum(int(ab.numel()) for *_,ab in binfo)
    need=int(20/100*totblk)
    thr=torch.kthvalue(pool,min(need,pool.numel())).values.item()
    for (Ro,Co,sc,ab),msk in zip(binfo,masks):
        zb=((sc.sqrt()<=thr)&ab).float(); msk.mul_(1.-(Ro@zb@Co.t()))
    for l,msk in zip(lays,masks):
        with torch.no_grad(): l.weight.mul_(emask(l,msk))
    a_cut=acc(mb); P('B[%s] acc right after 20%% cut (no finetune): %.2f'%(tag,a_cut))
    opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ft_epochs*len(trl))
    for e in range(ft_epochs):
        mb.train()
        for x,y in trl:
            x,y=x.to(dev),y.to(dev)
            loss=F.cross_entropy(net(x),y)+net.get_cost()
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            with torch.no_grad():
                for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
        P('B[%s] ft epoch %d done'%(tag,e))
    a_ft=acc(mb)
    zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
    tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
    P('B[%s] FINAL: blk-sp %.1f%%  acc %.2f'%(tag,100.*zb/tb,a_ft))
    del mb, net; torch.cuda.empty_cache()
    return a_cut, a_ft

r1=run('control mag-score 2ep/20k', 'mag',    2, False)   # expect ~83.3
r2=run('taylor-score      2ep/20k', 'taylor', 2, False)
r3=run('mag-score  LONG   6ep/50k', 'mag',    6, True)
P('SUMMARY  control=%.2f/%.2f  taylor=%.2f/%.2f  longft=%.2f/%.2f'%(r1+r2+r3))
