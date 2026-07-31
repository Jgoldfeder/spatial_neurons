import numpy as np,torch,torch.nn as nn,torchvision,sys,warnings,pickle
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
def mk(path):
    base=torchvision.models.resnet18(); base.fc=nn.Linear(512,100)
    net=swc.SpatialCNN(base,gamma=1.,device=dev,block_size=B).to(dev)
    net.model.load_state_dict(torch.load(path,map_location=dev)); return net
def set_trained_pos(net,pospath):
    for p,s in zip(net.pos,torch.load(pospath,map_location=dev)['pos']): p.copy_(s.to(dev))
def onehots(net):
    D=[]
    for l in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(l)
        def ids(x,y):
            k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go=ids(xo,yo); gi=ids(xi,yi); o=len(go); i=len(gi)
        Ro=torch.zeros(o,go.max()+1,device=dev); Ro[torch.arange(o),torch.tensor(go,device=dev)]=1
        Co=torch.zeros(i,gi.max()+1,device=dev); Co[torch.arange(i),torch.tensor(gi,device=dev)]=1
        D.append((Ro,Co))
    return D
LEVELS=[0,50,70,80,85,90]
def blockprune_curve(net):
    lays=net.layers; D=onehots(net); orig=[l.weight.detach().clone() for l in lays]
    bsq=[]; alln=[]
    for l,(Ro,Co) in zip(lays,D):
        W=wmat(l); b=Ro.t()@(W*W)@Co; bsq.append(b); alln.append(b.flatten())
    alln=torch.cat(alln).clamp(min=0).sqrt()
    out=[]
    for p in LEVELS:
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
        if p>0:
            thr2=torch.quantile(alln,p/100.).item()**2
            for l,(Ro,Co),b in zip(lays,D,bsq):
                z=(b<=thr2).float(); wmat(l).mul_(1.0-(Ro@z@Co.t()))
        out.append(acc(net.model))
    for l,w0 in zip(lays,orig): l.weight.copy_(w0)
    return out
# --- model sets (matched-ish base ~67-73) ---
import glob
MODELS={
 'spatial_swap':[('blk_models/swap_%g_ep14.pt'%g,'blk_models/swap_%g_ep14_pos.pt'%g) for g in [128,256,384]],
 'spatial_noswap':[('blk_models/noswap_%g_ep14.pt'%g,None) for g in [64,128,256]],
 'L1':[('rn_regime2_models/l1_%g_ep%02d.pt'%(g,max(e for (m,gg,e) in __import__('pickle').load(open('rn_regime2_shard0.pkl','rb'))['rec'] if False)),None) for g in []],
}
# L1: pick final epochs from disk
def l1_final(g):
    cs=[c for c in glob.glob('rn_regime2_models/l1_%g_ep*.pt'%g) if 'pos' not in c]
    e=max(int(c.rsplit('_ep',1)[1].split('.')[0]) for c in cs); return 'rn_regime2_models/l1_%g_ep%02d.pt'%(g,e)
MODELS['L1']=[(l1_final(g),None) for g in [1000,1500,2000]]
res={}
for cond,lst in MODELS.items():
    res[cond]=[]
    for path,pos in lst:
        net=mk(path)
        if pos: set_trained_pos(net,pos)                 # swap: trained tiling
        else:   net.swap(block=256)                      # PROPER reorder for L1/noswap
        cur=blockprune_curve(net)
        res[cond].append((cur[0],cur))                   # (base, curve over LEVELS)
        print('%-14s %-34s base %.1f | %s'%(cond,path.split('/')[-1],cur[0],
              '  '.join('%d%%:%.1f'%(LEVELS[i],cur[i]) for i in range(len(LEVELS)))),flush=True)
pickle.dump({'res':res,'LEVELS':LEVELS},open('/home/judah/spatial_v2/block_sparsity/blk3way.pkl','wb'))
print('done')
