import pickle,numpy as np,torch,torch.nn as nn,torchvision,sys,warnings
warnings.filterwarnings('ignore'); torch.set_grad_enabled(False)
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
B=16; dev='cuda'
from torch.utils.data import DataLoader
pre=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=pre),512,num_workers=4)
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
def load(g):
    base=torchvision.models.resnet18(); base.fc=nn.Linear(512,100)
    net=swc.SpatialCNN(base,gamma=float(g),device=dev,block_size=B).to(dev); m=net.model
    m.load_state_dict(torch.load('blk_models/swap_%g_ep14.pt'%g,map_location=dev))
    for p,s in zip(net.pos,torch.load('blk_models/swap_%g_ep14_pos.pt'%g,map_location=dev)['pos']): p.copy_(s.to(dev))
    return net,m
def onehots(net):
    D=[]
    for l in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(l)
        def ids(x,y):
            k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go=ids(xo,yo); gi=ids(xi,yi)
        o=len(go); i=len(gi); no=go.max()+1; ni=gi.max()+1
        Ro=torch.zeros(o,no,device=dev); Ro[torch.arange(o),torch.tensor(go,device=dev)]=1
        Co=torch.zeros(i,ni,device=dev); Co[torch.arange(i),torch.tensor(gi,device=dev)]=1
        D.append((Ro,Co))
    return D
for g in [128,256,384]:
    net,m=load(g); lays=[x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]; D=onehots(net)
    orig=[l.weight.detach().clone() for l in lays]
    bsq=[]  # per-layer (no,ni) squared block norms
    alln=[]
    for l,(Ro,Co) in zip(lays,D):
        W=wmat(l); b=Ro.t()@(W*W)@Co; bsq.append(b); alln.append(b.flatten())
    alln=torch.cat(alln).clamp(min=0).sqrt()
    print('swap g=%d: base %.1f  (%d blocks)'%(g,acc(m),alln.numel()),flush=True)
    print('  blocks-gone   acc',flush=True)
    for p in [0,50,70,80,85,90,95]:
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
        if p>0:
            thr2=torch.quantile(alln,p/100.).item()**2
            for l,(Ro,Co),b in zip(lays,D,bsq):
                zmask=(b<=thr2).float(); zelem=Ro@zmask@Co.t()
                wmat(l).mul_(1.0-zelem)
        print('     %2d%%       %5.1f'%(p,acc(m)),flush=True)
    for l,w0 in zip(lays,orig): l.weight.copy_(w0)
print('done',flush=True)
