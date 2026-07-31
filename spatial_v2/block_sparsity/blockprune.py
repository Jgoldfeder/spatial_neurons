import pickle,glob,numpy as np,torch,torch.nn as nn,torchvision,sys,warnings
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
def groups(net):
    G=[]
    for l in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(l)
        def ids(x,y):
            k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        G.append((ids(xo,yo),ids(xi,yi)))
    return G
for g in [128,256,384]:
    net,m=load(g); lays=[x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]; G=groups(net)
    orig=[l.weight.detach().clone() for l in lays]
    # collect all block norms (global) on trained tiling
    binfo=[]  # (layer_idx, out_site, in_site, norm)
    for li,(l,(go,gi)) in enumerate(zip(lays,G)):
        W=wmat(l).detach(); 
        for a in np.unique(go):
            ra=np.where(go==a)[0]
            for b in np.unique(gi):
                cb=np.where(gi==b)[0]
                nrm=W[np.ix_(ra,cb)].norm().item(); binfo.append((li,ra,cb,nrm))
    norms=np.array([x[3] for x in binfo])
    print('swap g=%d: base acc %.1f  (%d blocks total)'%(g,acc(m),len(binfo)))
    print('  blocks-gone%%   acc')
    for p in [0,50,70,80,85,90,95]:
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)   # restore
        if p>0:
            thr=np.quantile(norms,p/100.)
            # zero blocks with norm below thr
            byl={}
            for (li,ra,cb,nrm) in binfo:
                if nrm<=thr: byl.setdefault(li,[]).append((ra,cb))
            for li,blks in byl.items():
                Wv=wmat(lays[li])
                for ra,cb in blks: Wv[np.ix_(ra,cb)]=0
        print('    %2d%%        %5.1f'%(p,acc(m)))
    for l,w0 in zip(lays,orig): l.weight.copy_(w0)
print('done')
