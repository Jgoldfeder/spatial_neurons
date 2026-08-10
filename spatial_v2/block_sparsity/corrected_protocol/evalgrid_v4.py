import sys, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'/home/judah/spatial_neurons')
import spatial_wrapper_cnn as swc
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

# v4 eval grid: one zoo model per invocation. For each tiling in {contig,swap,tetris}
# x criterion {mag,taylor} x target {80,90,95,97,98,99}: one-shot cut -> post-cut acc
# -> 1 epoch plain finetune -> acc. All cells independent (weights restored each time).
ARCH, METHOD, HP = sys.argv[1], sys.argv[2], float(sys.argv[3])
dev='cuda'; B=64; torch.manual_seed(0); np.random.seed(0)
OUT=os.environ.get('SN_OUT','/home/judah/spatial_neurons')
ZD=OUT+'/v4_zoo'; TAG='%s_%s_%g'%(ARCH,METHOD,HP)
TARGETS=[80,90,95,97,98,99]; TNB=50

tr=T.Compose([T.Resize(224),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(224),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
trl=DataLoader(torchvision.datasets.CIFAR100(OUT+'/data',train=True,transform=tr),64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100(OUT+'/data',train=False,transform=te),256,num_workers=4)

def base_arch():
    if ARCH=='vit':
        import timm; return timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
    if ARCH=='rn50':
        m=torchvision.models.resnet50(); m.fc=nn.Linear(2048,100); return m
    m=torchvision.models.resnet18(); m.fc=nn.Linear(512,100); return m
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.weight.shape[0],-1) if isinstance(l,nn.Conv2d) else l.weight
def gmat(l): return l.weight.grad.view(wmat(l).shape)
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
@torch.no_grad()
def acc(m):
    m.eval(); c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100.*c/t
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.as_tensor(ids,device=dev,dtype=torch.long)]=1.; return O

def tiling_contig(lays):
    D=[]
    for l in lays:
        o,i=wmat(l).shape
        D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(i)//B,int(np.ceil(i/B)))))
    return D
def tiling_swap(mb):
    net=swc.SpatialCNN(mb,gamma=64.,device=dev,block_size=B).to(dev)
    pf='%s/%s_pos.pt'%(ZD,TAG)
    if os.path.exists(pf):
        pos=torch.load(pf,map_location='cpu')['pos']
        with torch.no_grad():
            for p,q in zip(net.pos,pos): p.copy_(q.to(dev))
    else:
        net.swap(block=256)
    D=[]
    for li in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(li)
        def ids(x,y):
            k=np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1)
            _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go,gi=ids(xo,yo),ids(xi,yi)
        D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    del net; return D
def balanced_groups(feats, gsize, iters=5):
    # capacity-constrained k-means: groups of exactly gsize (last may be smaller)
    n=feats.shape[0]; k=int(np.ceil(n/gsize))
    idx=torch.randperm(n)[:k]; C=feats[idx].clone()
    asg=torch.zeros(n,dtype=torch.long)
    for _ in range(iters):
        d=torch.cdist(feats,C)                     # (n,k)
        order=torch.argsort(d.min(1).values)       # confident rows first
        cap=torch.full((k,),gsize,dtype=torch.long); cap[-1]=n-(k-1)*gsize
        for r in order.tolist():
            pref=torch.argsort(d[r])
            for c in pref.tolist():
                if cap[c]>0: asg[r]=c; cap[c]-=1; break
        for c in range(k):
            m=asg==c
            if m.any(): C[c]=feats[m].mean(0)
    return asg.numpy()
def tiling_tetris(lays):
    # TETRIS-style post-hoc: alternating balanced co-clustering of rows/cols by mass profile
    D=[]
    for l in lays:
        W=wmat(l).detach().abs().cpu()
        o,i=W.shape
        ci=np.arange(i)//B
        for _ in range(3):
            nc=int(np.ceil(i/B))
            Rf=torch.zeros(o,nc)
            for c in range(nc): Rf[:,c]=W[:,ci==c].sum(1)
            ro=balanced_groups(Rf/ (Rf.norm(dim=1,keepdim=True)+1e-9), B)
            nr=int(np.ceil(o/B))
            Cf=torch.zeros(i,nr)
            for r in range(nr): Cf[:,r]=W[ro==r].sum(0)
            ci=balanced_groups(Cf/(Cf.norm(dim=1,keepdim=True)+1e-9), B)
        D.append((onehot(ro,int(ro.max()+1)),onehot(ci,int(ci.max()+1))))
    return D

# ---- load model, per-weight taylor importance once ----
sd=torch.load('%s/%s.pt'%(ZD,TAG),map_location='cpu')
mb=base_arch(); mb.load_state_dict(sd); mb=mb.to(dev).eval()
lays=regl(mb); orig=[l.weight.detach().clone() for l in lays]
dense=acc(mb)
timp=[torch.zeros_like(wmat(l)) for l in lays]
n=0
for i,(x,y) in enumerate(trl):
    mb.zero_grad(); F.cross_entropy(mb(x.to(dev)),y.to(dev)).backward()
    for j,l in enumerate(lays): timp[j]+=(gmat(l)*wmat(l).detach())**2
    n+=1
    if i+1>=TNB: break
mb.zero_grad(); timp=[t/n for t in timp]
print('[%s] dense %.2f | taylor imp done'%(TAG,dense),flush=True)

TIL={'contig':tiling_contig(lays),'swap':tiling_swap(mb),'tetris':tiling_tetris(lays)}
res={}
for tname,D in TIL.items():
    for crit in ['mag','taylor']:
        scores=[]
        for l,t_,(Ro,Co) in zip(lays,timp,D):
            base=(wmat(l).detach()**2) if crit=='mag' else t_
            s=(Ro.t()@base@Co)
            scores.append(s/(s.pow(2).sum().sqrt()+1e-12))   # per-layer l2 norm
        allsc=torch.cat([s.flatten() for s in scores])
        for tgt in TARGETS:
            k=int(tgt/100*allsc.numel())
            thr=torch.kthvalue(allsc,max(k,1)).values
            with torch.no_grad():
                for l,s,(Ro,Co),w0 in zip(lays,scores,D,orig):
                    zb=(s<=thr).float()
                    l.weight.copy_(w0*emask(l,1.-(Ro@zb@Co.t())).clamp(0,1))
            a0=acc(mb)
            masks=[(wmat(l)!=0).float() for l in lays]
            opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
            mb.train()
            for x,y in trl:
                x,y=x.to(dev),y.to(dev)
                F.cross_entropy(mb(x),y).backward(); opt.step(); opt.zero_grad()
                with torch.no_grad():
                    for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
            a1=acc(mb)
            with torch.no_grad():
                for l,w0 in zip(lays,orig): l.weight.copy_(w0)
            res[(tname,crit,tgt)]=(a0,a1)
            print('[%s] %s/%s tgt %d | post-cut %.2f | ft1 %.2f'%(TAG,tname,crit,tgt,a0,a1),flush=True)
import pickle; pickle.dump({'dense':dense,'res':res},open('%s/%s_grid.pkl'%(ZD,TAG),'wb'))
print('done',flush=True)
