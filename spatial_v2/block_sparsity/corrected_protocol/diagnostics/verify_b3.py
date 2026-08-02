import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import spatial_wrapper_cnn as swc
import timm, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

dev='cuda'; B=16; R=224
t0=time.time(); P=lambda s: print('[%7.1fs] %s'%(time.time()-t0,s), flush=True)

tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr)
trl=DataLoader(full,64,shuffle=True,num_workers=8,pin_memory=True)      # FULL 50k
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)

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

torch.manual_seed(0); np.random.seed(0)
mb=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
mb.load_state_dict(torch.load('../vitbase_cifar100_base.pt',map_location='cpu')); mb=mb.to(dev)
net=swc.SpatialCNN(mb,gamma=64.,device=dev,block_size=B).to(dev)
P('swap start'); net.swap(block=256); P('swap done')
D=[]
for l in range(len(net.layers)):
    xi,yi,xo,yo=net.planes(l)
    def ids(x,y):
        k=np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1)
        _,inv=np.unique(k,axis=0,return_inverse=True); return inv
    go,gi=ids(xo,yo),ids(xi,yi); D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
lays=regl(mb)
orig=[l.weight.detach().clone() for l in lays]

def cut(scoring):
    masks=[torch.ones_like(wmat(l)) for l in lays]
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
    thr=torch.kthvalue(pool,min(int(0.20*totblk),pool.numel())).values.item()
    for (Ro,Co,sc,ab),msk in zip(binfo,masks):
        zb=((sc.sqrt()<=thr)&ab).float(); msk.mul_(1.-(Ro@zb@Co.t()))
    for l,msk in zip(lays,masks):
        with torch.no_grad(): l.weight.mul_(emask(l,msk))
    return masks

# ---- taylor post-cut only (no finetune) ----
m_t=cut('taylor'); P('TAYLOR post-cut (no ft): %.2f'%acc(mb))
with torch.no_grad():
    for l,w0 in zip(lays,orig): l.weight.copy_(w0)

# ---- B3: mag cut + LONG finetune 6ep x 50k ----
masks=cut('mag'); P('MAG post-cut (no ft): %.2f'%acc(mb))
opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6*len(trl))
for e in range(6):
    mb.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev)
        loss=F.cross_entropy(net(x),y)+net.get_cost()
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        with torch.no_grad():
            for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
    P('B3 ft epoch %d done, acc %.2f'%(e,acc(mb)))
P('B3 FINAL (mag cut, 6ep x 50k): %.2f   [control was 83.31 with 2ep x 20k]'%acc(mb))
