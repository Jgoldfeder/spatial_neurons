import pickle,glob,numpy as np,torch,torch.nn as nn,torchvision,sys,warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
B=16; dev='cuda'
from torch.utils.data import DataLoader
pre=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=pre),512,num_workers=4)
base=torchvision.models.resnet18(); base.fc=nn.Linear(512,100)
net=swc.SpatialCNN(base,gamma=256.,device=dev,block_size=B).to(dev); m=net.model
m.load_state_dict(torch.load('blk_models/swap_256_ep14.pt',map_location=dev))
for p,s in zip(net.pos,torch.load('blk_models/swap_256_ep14_pos.pt',map_location=dev)['pos']): p.copy_(s.to(dev))
lays=[x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
# trained-tiling group ids
gids=[]
for l in range(len(net.layers)):
    xi,yi,xo,yo=net.planes(l)
    def ids(x,y):
        k=np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
    gids.append((ids(xo,yo),ids(xi,yi)))
@torch.no_grad()
def acc():
    m.eval(); c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100.*c/t
def emptyblk(thr):
    num=den=0
    for l,(go,gi) in zip(lays,gids):
        W=wmat(l).abs().detach().cpu().numpy(); o,i=W.shape
        if o<B or i<B: continue
        oo=np.argsort(go,kind='stable'); ii=np.argsort(gi,kind='stable'); M=W[oo][:,ii]; go2=go[oo]; gi2=gi[ii]
        ob=np.append(np.searchsorted(go2,np.arange(go2.max()+1)),len(go2)); ib=np.append(np.searchsorted(gi2,np.arange(gi2.max()+1)),len(gi2))
        al=M>=thr
        for a in range(len(ob)-1):
            ra=al[ob[a]:ob[a+1]]
            if ra.size==0: continue
            for b in range(len(ib)-1):
                blk=ra[:,ib[b]:ib[b+1]]
                if blk.size: num+=(not blk.any()); den+=1
    return 100.*num/max(den,1)
import torch as _t
_t.set_grad_enabled(False)
orig=[l.weight.detach().clone() for l in lays]
allw=torch.cat([w.abs().flatten() for w in orig]).cpu().numpy()
print('swap g=256 FINE grid (trained tiling):  sp%  acc   blk')
for s in [88,90,91,92,93,94,95]:
    thr=float(np.quantile(allw,s/100.))
    for l,w0 in zip(lays,orig): l.weight.copy_(torch.where(w0.abs()<thr,torch.zeros_like(w0),w0))
    print('   %2d%%  %5.1f  %5.1f'%(s,acc(),emptyblk(thr)))
    for l,w0 in zip(lays,orig): l.weight.copy_(w0)
print('done')
