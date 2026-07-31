import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import timm,torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader,Subset
GAMMAS=[float(g) for g in sys.argv[1].split(',')]
dev='cuda'; B=15; R=224; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr); idx=np.random.RandomState(0).permutation(len(full))[:12000]
trl=DataLoader(Subset(full,idx),64,shuffle=True,num_workers=3,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=2)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
def swap_tiling(net):
    D=[]
    for l in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(l)
        def ids(x,y):
            k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go,gi=ids(xo,yo),ids(xi,yi); D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D
LEV=[0,20,40,55,65,75,85,90]; t0=time.time()
res={}
for GAMMA in GAMMAS:
    torch.manual_seed(0); np.random.seed(0)
    base=timm.create_model('vit_small_patch16_224',pretrained=True,num_classes=100).to(dev)
    net=swc.SpatialCNN(base,gamma=GAMMA,device=dev,block_size=B).to(dev); m=net.model
    opt=torch.optim.AdamW([{'params':m.parameters(),'lr':1e-4}],weight_decay=0.05)
    for e in range(4):                       # 4-epoch spatial finetune
        m.train(); net.swap(block=256)
        for x,y in trl:
            x,y=x.to(dev),y.to(dev); (F.cross_entropy(net(x),y)+net.get_cost()).backward(); opt.step(); opt.zero_grad()
    net.swap(block=256); D=swap_tiling(net); lays=regl(m)
    base_acc=acc(m)
    # block-importance prune curve on the swap tiling (block=15)
    orig=[l.weight.detach().clone() for l in lays]
    bsq=[]; alln=[]
    for l,(Ro,Co) in zip(lays,D):
        W=wmat(l); b=(Ro.t()@(W*W)@Co); bsq.append(b); alln.append(b.flatten())
    alln=torch.cat(alln).clamp(min=0).sqrt()
    curve=[]
    for p in LEV:
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
        if p>0:
            thr=torch.quantile(alln,p/100.).item()**2
            for l,(Ro,Co),b in zip(lays,D,bsq):
                z=(b<=thr).float(); wmat(l).mul_(1.-(Ro@z@Co.t()))
        curve.append((p,acc(m)))
    for l,w0 in zip(lays,orig): l.weight.copy_(w0)
    res[GAMMA]={'base':base_acc,'curve':curve}
    print('[vits15 g=%g] base %.1f | '%(GAMMA,base_acc)+' '.join('%d:%.1f'%(p,a) for p,a in curve)+' (%.0fs)'%(time.time()-t0),flush=True)
    import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/vitsmall_b15_%s.pkl'%('_'.join('%g'%g for g in GAMMAS)),'wb'))
print('done',flush=True)
