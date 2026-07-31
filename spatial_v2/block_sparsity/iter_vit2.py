import sys,time,copy,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import timm,torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader,Subset
METHOD=sys.argv[1]        # mag|spatial|glasso|taylor|spataylor
TIL=sys.argv[2]           # contig|reorder
HP=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
dev='cuda'; B=16; R=224; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr); idx=np.random.RandomState(0).permutation(len(full))[:20000]
trl=DataLoader(Subset(full,idx),64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)
def vit():
    m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
    m.load_state_dict(torch.load('/home/judah/spatial_v2/block_sparsity/vitbase_cifar100_base.pt',map_location='cpu')); return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def gmat(l): return l.weight.grad.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight.grad
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
def contig_tiling(m):
    D=[]
    for l in regl(m):
        o,i=wmat(l).shape; D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(i)//B,int(np.ceil(i/B)))))
    return D
def swap_tiling(net):
    D=[]
    for l in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(l)
        def ids(x,y):
            k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go,gi=ids(xo,yo),ids(xi,yi); D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D
def set_contig_pos(net,side=20.):
    for l in range(len(net.layers)):
        n_in,n_out,_,_=net.meta[l]
        for bi,n in [(4*l,n_in),(4*l+2,n_out)]:
            ns=max(1,math.ceil(n/B)); rows=int(math.ceil(math.sqrt(ns))); cols=int(math.ceil(ns/rows))
            xs=torch.linspace(-side/2,side/2,cols); ys=torch.linspace(-side/2,side/2,rows)
            gx,gy=torch.meshgrid(xs,ys,indexing='xy'); sx=gx.flatten()[:ns]; sy=gy.flatten()[:ns]
            site=torch.clamp(torch.arange(n)//B,max=ns-1)
            with torch.no_grad(): net.pos[bi].copy_(sx[site].to(dev)); net.pos[bi+1].copy_(sy[site].to(dev))
PRUNE_TAYLOR=METHOD in ('taylor','spataylor'); SPATIAL_COST=METHOD in ('spatial','spataylor')
LEV=[20,40,55,65,75,80,85,90]; F_ep=2; t0=time.time()
mb=vit(); print('base %.1f (%.0fs)'%(acc(mb),time.time()-t0),flush=True)
net=None
if TIL=='reorder':
    net=swc.SpatialCNN(mb,gamma=(HP if SPATIAL_COST else 64.),device=dev,block_size=B).to(dev); net.swap(block=256); D=swap_tiling(net)
else:  # contig -- if spatial cost needed, set contiguous positions so cost aligns with contig blocks
    if SPATIAL_COST:
        net=swc.SpatialCNN(mb,gamma=HP,device=dev,block_size=B).to(dev); set_contig_pos(net)
    D=contig_tiling(mb)
usenet=net if SPATIAL_COST else None
lays=regl(mb); masks=[torch.ones_like(wmat(l)) for l in lays]; out=[]
for target in LEV:
    if PRUNE_TAYLOR:
        mb.zero_grad()
        for i,(x,y) in enumerate(trl):
            F.cross_entropy((usenet(x.to(dev)) if usenet else mb(x.to(dev))),y.to(dev)).backward()
            if i>=10: break
    alln=[]; binfo=[]
    for l,(Ro,Co),msk in zip(lays,D,masks):
        W=wmat(l)*msk
        sc=(Ro.t()@((gmat(l)*W)**2)@Co) if PRUNE_TAYLOR else (Ro.t()@(W*W)@Co)
        ab=(Ro.t()@msk@Co)>0; binfo.append((Ro,Co,sc,ab)); alln.append(sc[ab].flatten())
    mb.zero_grad()
    pool=torch.cat(alln).clamp(min=0).sqrt()
    totblk=sum(int(ab.numel()) for *_,ab in binfo); already=sum(int((~ab).sum()) for *_,ab in binfo)
    need=max(0,int(target/100*totblk)-already)
    if need>0 and pool.numel()>0:
        thr=torch.kthvalue(pool,min(need,pool.numel())).values.item()
        for (Ro,Co,sc,ab),msk in zip(binfo,masks):
            zb=((sc.sqrt()<=thr)&ab).float(); msk.mul_(1.-(Ro@zb@Co.t()))
    for l,msk in zip(lays,masks):
        with torch.no_grad(): l.weight.mul_(emask(l,msk))
    opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,F_ep*len(trl))
    for e in range(F_ep):
        mb.train()
        for x,y in trl:
            x,y=x.to(dev),y.to(dev); loss=F.cross_entropy((usenet(x) if usenet else mb(x)),y)
            if usenet is not None: loss=loss+usenet.get_cost()
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            with torch.no_grad():
                for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
    zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
    tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
    a=acc(mb); out.append((100.*zb/tb,a)); print('[vit2-%s-%s g%g] tgt %2d%% | blk-sp %4.1f%% | acc %.1f (%.0fs)'%(METHOD,TIL,HP,target,100.*zb/tb,a,time.time()-t0),flush=True)
import pickle; pickle.dump(out,open('/home/judah/spatial_v2/block_sparsity/vit2_%s_%s_%g.pkl'%(METHOD,TIL,HP),'wb'))
print('done',flush=True)
