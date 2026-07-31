import sys,time,copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader
TIL=sys.argv[1]  # 'contig' or 'perm'
dev='cuda'; B=16; R=160; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
trl=DataLoader(torchvision.datasets.CIFAR100('./data',train=True,transform=tr),96,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),512,num_workers=4)
def rn18():
    m=torchvision.models.resnet18(weights='IMAGENET1K_V1'); m.fc=nn.Linear(512,100); return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def gmat(l): return l.weight.grad.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight.grad
def emask(l,msk): return msk.view_as(l.weight) if isinstance(l,nn.Conv2d) else msk
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
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
        else: go=np.arange(o)//B; gi=np.arange(inn)//B
        D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D
def iterative_move(m,D,levels,F_ep,tag,net=None):
    lays=regl(m); masks=[torch.ones_like(wmat(l)) for l in lays]; out=[]
    for target in levels:
        # --- 1 scoring epoch: finetune + accumulate MOVEMENT score (-grad*W) on alive weights ---
        scores=[torch.zeros_like(wmat(l)) for l in lays]; opt=torch.optim.Adam(m.parameters(),5e-4)
        m.train()
        for x,y in trl:
            x,y=x.to(dev),y.to(dev); (F.cross_entropy((net(x) if net else m(x)),y)+(net.get_cost() if net else 0)).backward()
            with torch.no_grad():
                for l,s,msk in zip(lays,scores,masks): s.add_(((gmat(l)*wmat(l))**2)*msk)
            opt.step(); opt.zero_grad()
            with torch.no_grad():
                for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
        # --- prune lowest block-MOVEMENT blocks to target ---
        alln=[]; binfo=[]
        for l,(Ro,Co),s,msk in zip(lays,D,scores,masks):
            bs=(Ro.t()@s@Co); ab=(Ro.t()@msk@Co)>0; binfo.append((Ro,Co,bs,ab)); alln.append(bs[ab].flatten())
        pool=torch.cat(alln)
        totblk=sum(int(ab.numel()) for *_,ab in binfo); already=sum(int((~ab).sum()) for *_,ab in binfo)
        need=max(0,int(target/100*totblk)-already)
        if need>0 and pool.numel()>0:
            thr=torch.kthvalue(pool,min(need,pool.numel())).values.item()
            for (Ro,Co,bs,ab),msk in zip(binfo,masks):
                zb=((bs<=thr)&ab).float(); msk.mul_(1.-(Ro@zb@Co.t()))
        for l,msk in zip(lays,masks):
            with torch.no_grad(): l.weight.mul_(emask(l,msk))
        # --- recover finetune (F_ep-1 epochs) ---
        opt=torch.optim.Adam(m.parameters(),5e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(1,(F_ep-1))*len(trl))
        for e in range(F_ep-1):
            m.train()
            for x,y in trl:
                x,y=x.to(dev),y.to(dev); (F.cross_entropy((net(x) if net else m(x)),y)+(net.get_cost() if net else 0)).backward(); opt.step(); sched.step(); opt.zero_grad()
                with torch.no_grad():
                    for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
        zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
        tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
        a=acc(m); out.append((100.*zb/tb,a)); print('[%s] tgt %2d%% | blk-sp %4.1f%% | acc %.1f'%(tag,target,100.*zb/tb,a),flush=True)
    return out
LEV=[20,40,55,65,75,80,85,90]; t0=time.time()
mb=rn18(); opt=torch.optim.Adam(mb.parameters(),1e-4)
for e in range(4):
    mb.train()
    for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(mb(x),y).backward(); opt.step(); opt.zero_grad()
print('base %.1f (%.0fs)'%(acc(mb),time.time()-t0),flush=True)
NET=None
if TIL in ('perm','spat'):
    NET=swc.SpatialCNN(mb,gamma=64.,device=dev,block_size=B).to(dev); NET.swap(block=256); D=tiling(mb,NET)
else: D=tiling(mb,None)
res=iterative_move(mb,D,LEV,2,'taylor-'+TIL, net=(NET if TIL=='spat' else None))   # spat: spatial cost ON during finetune
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/iter_move_%s.pkl'%TIL,'wb'))
print('done (%.0fs)'%(time.time()-t0))
