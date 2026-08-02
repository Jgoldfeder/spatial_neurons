import sys,time,copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import spatial_wrapper_cnn as swc
import timm,torchvision,torchvision.transforms as T
from torch.utils.data import DataLoader,Subset
METHOD=sys.argv[1]        # mag | spatial | glasso | taylor
TIL=sys.argv[2]           # contig | reorder
HP=float(sys.argv[3]) if len(sys.argv)>3 else 0.0   # gamma(spatial) / lambda(glasso)
VARIANT=sys.argv[4] if len(sys.argv)>4 else ''      # '' | polish (gamma off last 2 ft epochs) | taylorscore (taylor criterion)
dev='cuda'; B=16; R=224; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr)
trl=DataLoader(full,64,shuffle=True,num_workers=8,pin_memory=True)   # LONG budget: full 50k
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)
def vit():
    m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
    m.load_state_dict(torch.load('./vitbase_cifar100_base.pt',map_location='cpu')); return m.to(dev)
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
def tiling(m,net=None):
    D=[]; lays=regl(m)
    if net is None:
        for l in lays:
            o,i=wmat(l).shape; D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(i)//B,int(np.ceil(i/B)))))
    else:
        for l in range(len(net.layers)):
            xi,yi,xo,yo=net.planes(l)
            def ids(x,y):
                k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
            go,gi=ids(xo,yo),ids(xi,yi); D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D
def gl_pen(lays,D):
    tot=0.;n=0
    for l,(Ro,Co) in zip(lays,D):
        bn=torch.sqrt((Ro.t()@(wmat(l)**2)@Co)+1e-12); tot=tot+bn.sum(); n+=bn.numel()
    return tot/n
LEV=[20,40,55,65,75,80,85,90,95,97,98,99]; F_ep=6; t0=time.time()
import os; CKDIR='./vitlong_models'; os.makedirs(CKDIR,exist_ok=True)
mb=vit(); print('base %.1f (%.0fs)'%(acc(mb),time.time()-t0),flush=True)
# tiling + optional spatial net
net=None
if METHOD=='spatial' or TIL=='reorder':
    net=swc.SpatialCNN(mb,gamma=(HP if METHOD=='spatial' else 64.),device=dev,block_size=B).to(dev); net.swap(block=256)
    D=tiling(mb,net)
else:
    D=tiling(mb,None)
usenet = net if METHOD=='spatial' else None      # spatial cost only for spatial arm
# ---- PRE-ORGANIZATION (spatial arm only): dense training WITH wiring cost BEFORE the first cut ----
if METHOD=='spatial':
    opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
    for e in range(6):
        mb.train()
        for x,y in trl:
            x,y=x.to(dev),y.to(dev)
            loss=F.cross_entropy(net(x),y)+net.get_cost()
            loss.backward(); opt.step(); opt.zero_grad()
        print('[vit-%s-%s hp=%g] pre-org epoch %d | dense acc %.2f (%.0fs)'%(METHOD,TIL,HP,e,acc(mb),time.time()-t0),flush=True)
        net.swap(block=256)
    D=tiling(mb,net)   # re-tile on the organized layout
    import os as _os; _os.makedirs('./vitlong_models',exist_ok=True)
    torch.save(mb.state_dict(),'./vitlong_models/%s%s_%s_%g_preorg.pt'%(METHOD,VARIANT,TIL,HP))
    torch.save({'pos':[p.detach().cpu() for p in net.pos]},'./vitlong_models/%s%s_%s_%g_pos.pt'%(METHOD,VARIANT,TIL,HP))
lays=regl(mb); masks=[torch.ones_like(wmat(l)) for l in lays]; out=[]
for target in LEV:
    # ---- score + prune ----
    if METHOD=='taylor' or VARIANT=='taylorscore':
        mb.zero_grad()
        for i,(x,y) in enumerate(trl):
            F.cross_entropy((usenet(x.to(dev)) if usenet else mb(x.to(dev))),y.to(dev)).backward()
            if i>=10: break
    alln=[]; binfo=[]
    for l,(Ro,Co),msk in zip(lays,D,masks):
        W=wmat(l)*msk
        if METHOD=='taylor' or VARIANT=='taylorscore': sc=(Ro.t()@((gmat(l)*W)**2)@Co)   # (gW)^2 = diagonal 2nd-order
        else:               sc=(Ro.t()@(W*W)@Co)                       # block L2
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
    print('[vit-%s-%s hp=%g] tgt %2d%% | post-cut (no ft) acc %.2f (%.0fs)'%(METHOD,TIL,HP,target,acc(mb),time.time()-t0),flush=True)
    # ---- finetune ----
    opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,F_ep*len(trl))
    for e in range(F_ep):
        mb.train()
        for x,y in trl:
            x,y=x.to(dev),y.to(dev)
            loss=F.cross_entropy((usenet(x) if usenet else mb(x)),y)
            if usenet is not None and not (VARIANT=='polish' and e>=F_ep-2): loss=loss+usenet.get_cost()   # polish: gamma off last 2 ft epochs
            if METHOD=='glasso': loss=loss+HP*gl_pen(lays,D)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            with torch.no_grad():
                for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
    zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
    tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
    a=acc(mb); out.append((100.*zb/tb,a)); print('[vit-%s-%s hp=%g] tgt %2d%% | blk-sp %4.1f%% | acc %.1f (%.0fs)'%(METHOD,TIL,HP,target,100.*zb/tb,a,time.time()-t0),flush=True)
    import pickle; pickle.dump(out,open('./vitlong_%s%s_%s_%g.pkl'%(METHOD,VARIANT,TIL,HP),'wb'))   # save per stage (crash-safe)
    torch.save(mb.state_dict(), '%s/%s%s_%s_%g_s%02d.pt'%(CKDIR,METHOD,VARIANT,TIL,HP,target))   # always save model weights
import pickle; pickle.dump(out,open('./vitlong_%s%s_%s_%g.pkl'%(METHOD,VARIANT,TIL,HP),'wb'))
print('done',flush=True)
