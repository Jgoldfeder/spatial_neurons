import sys,time,copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
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
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
def train(m,net,ep,swap,lr=1e-4):
    opt=torch.optim.Adam(m.parameters(),lr)
    for e in range(ep):
        m.train()
        if swap: net.swap(block=256)
        for x,y in trl:
            x,y=x.to(dev),y.to(dev)
            loss=F.cross_entropy(net(x) if net else m(x),y)+(net.get_cost() if net else 0)
            loss.backward(); opt.step(); opt.zero_grad()
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
def tiling(m,net=None):
    D=[]; lays=regl(m)
    for i,l in enumerate(lays):
        o,inn=wmat(l).shape
        if net is not None:
            xi,yi,xo,yo=net.planes(i)
            def ids(x,y):
                k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
            go,gi=ids(xo,yo),ids(xi,yi)
        else:
            go=np.arange(o)//B; gi=np.arange(inn)//B
        D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D
def emask(l,msk): return msk.view_as(l.weight) if isinstance(l,nn.Conv2d) else msk
def iterative(m,D,levels,F_ep,tag):
    lays=regl(m); masks=[torch.ones_like(wmat(l)) for l in lays]; out=[]
    for target in levels:
        alln=[]; binfo=[]
        for l,(Ro,Co),msk in zip(lays,D,masks):
            W=wmat(l)*msk; b=(Ro.t()@(W*W)@Co); ab=(Ro.t()@msk@Co)>0
            binfo.append((Ro,Co,b,ab)); alln.append(b[ab].flatten())
        pool=torch.cat(alln).clamp(min=0).sqrt()
        totblk=sum(int(ab.numel()) for *_,ab in binfo); already=sum(int((~ab).sum()) for *_,ab in binfo)
        need=max(0,int(target/100*totblk)-already)
        if need>0 and pool.numel()>0:
            thr=torch.kthvalue(pool,min(need,pool.numel())).values.item()
            for (Ro,Co,b,ab),msk in zip(binfo,masks):
                zb=((b.sqrt()<=thr)&ab).float(); msk.mul_(1.-(Ro@zb@Co.t()))
        for l,msk in zip(lays,masks):
            with torch.no_grad(): l.weight.mul_(emask(l,msk))
        opt=torch.optim.Adam(m.parameters(),5e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,F_ep*len(trl))
        for e in range(F_ep):
            m.train()
            for x,y in trl:
                x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); sched.step(); opt.zero_grad()
                with torch.no_grad():
                    for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
        zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
        tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
        a=acc(m); out.append((100.*zb/tb,a)); print('[%s] tgt %2d%% | blk-sp %4.1f%% | acc %.1f'%(tag,target,100.*zb/tb,a),flush=True)
    return out
LEV=[20,40,55,65,75,80,85,90]; t0=time.time()
print('=== base: baseline finetune ==='); mb=rn18(); train(mb,None,4,False); print('baseline base %.1f (%.0fs)'%(acc(mb),time.time()-t0),flush=True)
print('=== base: spatial (swap) finetune ==='); base=rn18(); net=swc.SpatialCNN(base,gamma=64.,device=dev,block_size=B).to(dev); train(net.model,net,4,True); print('spatial base %.1f (%.0fs)'%(acc(net.model),time.time()-t0),flush=True)
# permuted baseline = copy of baseline weights, proper swap-reorder tiling
mp=copy.deepcopy(mb); netp=swc.SpatialCNN(mp,gamma=64.,device=dev,block_size=B).to(dev); netp.swap(block=256)
res={}
print('=== iterative (full data, gradual, 2ep/level, cosine) ===')
res['baseline_contig']=iterative(mb,tiling(mb,None),LEV,2,'baseline-contig')
res['baseline_perm']=iterative(netp.model,tiling(netp.model,netp),LEV,2,'baseline-PERMUTED')
res['spatial_swap']=iterative(net.model,tiling(net.model,net),LEV,2,'spatial-swap')
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/iter_prune2.pkl','wb'))
print('done (%.0fs)'%(time.time()-t0))
