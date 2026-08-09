import sys, os, time, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'/home/judah/spatial_neurons')
import spatial_wrapper_cnn as swc
import timm, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

# ============================================================================
# v3: corrected + strengthened block-sparsity baselines. One script, all arms.
#   ARCH   : vit | rn50
#   METHOD : mag | taylor | spatial | glasso | movement | rigl
#   TIL    : reorder | contig
#   HP     : gamma (spatial) / lambda (glasso) / unused otherwise (pass 0)
#   VARIANT: '' | taylorscore (taylor criterion on a reg-trained model) | polish
# Fixes vs earlier suites:
#   * taylor: per-minibatch SQUARED accumulation E[(gW)^2] (no signed cancellation),
#             50 minibatches, per-layer l2-normalized scores before global ranking
#   * mag:    per-layer l2-normalized block scores (BN-scale robust)
#   * glasso: gets the same pre-organization phase as spatial (6 dense epochs with
#             the penalty) instead of only post-cut regularization
#   * movement: real block-movement baseline — learnable per-block scores, straight-
#             through top-v masking, cubic sparsity ramp inside each stage
#   * rigl:   real block-RigL — per-target dynamic sparse training from the base,
#             drop by block |W|, grow by block |g|, cosine-decayed update fraction
#   * all:    pre-org for regularizer methods, 6ep x full-50k finetune per stage,
#             per-stage crash-safe pkl + checkpoint saving, post-cut acc logged
# ============================================================================
ARCH   = sys.argv[1]
METHOD = sys.argv[2]
TIL    = sys.argv[3]
HP     = float(sys.argv[4]) if len(sys.argv)>4 else 0.0
VARIANT= sys.argv[5] if len(sys.argv)>5 else ''
SMOKE  = os.environ.get('SMOKE','')=='1'
F_ep   = int(os.environ.get('F_EP','6'))
PRE_ep = int(os.environ.get('PREORG_EP','6'))
TNB    = int(os.environ.get('TAYLOR_NB','50'))
LEV    = [20,40] if SMOKE else [20,40,55,65,75,80,85,90,95,97,98,99]
if SMOKE: F_ep=1; PRE_ep=1; TNB=5

dev='cuda'; B=64; R=224; torch.manual_seed(0); np.random.seed(0)
TAG='v3_%s_%s%s_%s_%g'%(ARCH,METHOD,VARIANT,TIL,HP)
OUT=os.environ.get('SN_OUT','/home/judah/spatial_neurons')   # data, base ckpts, results
CKDIR=OUT+'/v3_models'; os.makedirs(CKDIR,exist_ok=True)

tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100(OUT+'/data',train=True,transform=tr)
trl=DataLoader(full,64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100(OUT+'/data',train=False,transform=te),256,num_workers=4)

def base_model():
    if ARCH=='vit':
        m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
        m.load_state_dict(torch.load(OUT+'/vitbase_cifar100_base.pt',map_location='cpu'))
    else:
        m=torchvision.models.resnet50(); m.fc=nn.Linear(2048,100)
        m.load_state_dict(torch.load(OUT+'/rn50_cifar100_base.pt',map_location='cpu'))
    return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.weight.shape[0],-1) if isinstance(l,nn.Conv2d) else l.weight
def gmat(l): return l.weight.grad.view(wmat(l).shape)
def emask(l,M): return M.view_as(l.weight) if isinstance(l,nn.Conv2d) else M
@torch.no_grad()
def acc(m):
    m.eval(); c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
        if SMOKE and t>=2560: break
    return 100.*c/t
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
def tiling(m,net=None):
    D=[]
    if net is None:
        for l in regl(m):
            o,i=wmat(l).shape
            D.append((onehot(np.arange(o)//B,int(np.ceil(o/B))),onehot(np.arange(i)//B,int(np.ceil(i/B)))))
    else:
        for li in range(len(net.layers)):
            xi,yi,xo,yo=net.planes(li)
            def ids(x,y):
                k=np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1)
                _,inv=np.unique(k,axis=0,return_inverse=True); return inv
            go,gi=ids(xo,yo),ids(xi,yi)
            D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D
def gl_pen(lays,D):
    tot=0.;n=0
    for l,(Ro,Co) in zip(lays,D):
        bn=torch.sqrt((Ro.t()@(wmat(l)**2)@Co)+1e-12); tot=tot+bn.sum(); n+=bn.numel()
    return tot/n

# ---------------- scoring (strengthened) ----------------
def taylor_scores(mb, lays, D, usenet):
    # E[(g*W)^2]: square PER MINIBATCH, then average -> no signed cancellation
    sc=[torch.zeros(Ro.shape[1],Co.shape[1],device=dev) for Ro,Co in D]
    n=0
    for i,(x,y) in enumerate(trl):
        mb.zero_grad()
        F.cross_entropy((usenet(x.to(dev)) if usenet else mb(x.to(dev))),y.to(dev)).backward()
        for j,(l,(Ro,Co)) in enumerate(zip(lays,D)):
            sc[j]+=(Ro.t()@((gmat(l)*wmat(l).detach())**2)@Co)
        n+=1
        if i+1>=TNB: break
    mb.zero_grad()
    return [s/n for s in sc]
def mag_scores(lays, D, masks):
    return [(Ro.t()@((wmat(l)*msk)**2)@Co) for l,(Ro,Co),msk in zip(lays,D,masks)]
def layer_normalize(scores):
    # Molchanov-style per-layer l2 normalization before global ranking
    return [s/ (s.pow(2).sum().sqrt()+1e-12) for s in scores]
def prune_to(target, lays, D, masks, scores):
    binfo=[]; alln=[]
    for (Ro,Co),msk,sc in zip(D,masks,scores):
        ab=(Ro.t()@msk@Co)>0
        binfo.append((Ro,Co,sc,ab)); alln.append(sc[ab].flatten())
    pool=torch.cat(alln)
    totblk=sum(int(ab.numel()) for *_,ab in binfo); already=sum(int((~ab).sum()) for *_,ab in binfo)
    need=max(0,int(target/100*totblk)-already)
    if need>0 and pool.numel()>0:
        need=min(need,pool.numel()-1) if need<pool.numel() else pool.numel()
        vals,_=torch.sort(pool)
        thr=vals[need-1]
        cut=0
        for (Ro,Co,sc,ab),msk in zip(binfo,masks):
            zb=((sc<thr)&ab)
            # exact-count tie handling: fill remaining need from ties at thr
            room=need-cut-int(zb.sum())
            if room>0:
                ties=torch.nonzero((sc==thr)&ab&~zb)
                for t_ in ties[:room]: zb[t_[0],t_[1]]=True
            cut+=int(zb.sum())
            msk.mul_(1.-(Ro@zb.float()@Co.t()))
    for l,msk in zip(lays,masks):
        with torch.no_grad(): l.weight.mul_(emask(l,msk))

def finetune(mb, lays, masks, usenet, D, epochs, gl=False):
    opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs*len(trl))
    for e in range(epochs):
        mb.train()
        for bi,(x,y) in enumerate(trl):
            x,y=x.to(dev),y.to(dev)
            loss=F.cross_entropy((usenet(x) if usenet else mb(x)),y)
            if usenet is not None and not (VARIANT=='polish' and e>=epochs-2): loss=loss+usenet.get_cost()
            if gl: loss=loss+HP*gl_pen(lays,D)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            with torch.no_grad():
                for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
            if SMOKE and bi>=30: break

def blocksp(lays,D):
    zb=sum(int((~((Ro.t()@(wmat(l)!=0).float()@Co)>0)).sum()) for l,(Ro,Co) in zip(lays,D))
    tb=sum(int(Ro.shape[1]*Co.shape[1]) for Ro,Co in D)
    return 100.*zb/tb

t0=time.time(); P=lambda s: print('[%s] %s (%.0fs)'%(TAG,s,time.time()-t0),flush=True)
mb=base_model(); P('base %.1f'%acc(mb))
net=None
if METHOD=='spatial' or TIL=='reorder':
    net=swc.SpatialCNN(mb,gamma=(HP if METHOD=='spatial' else 64.),device=dev,block_size=B).to(dev); net.swap(block=256)
    D=tiling(mb,net)
else: D=tiling(mb,None)
usenet=net if METHOD=='spatial' else None
lays=regl(mb)

# ---------------- pre-organization for regularizer methods (spatial AND glasso) ----------------
if METHOD in ('spatial','glasso'):
    opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
    for e in range(PRE_ep):
        mb.train()
        for bi,(x,y) in enumerate(trl):
            x,y=x.to(dev),y.to(dev)
            loss=F.cross_entropy((net(x) if METHOD=='spatial' else mb(x)),y)
            loss=loss+(net.get_cost() if METHOD=='spatial' else HP*gl_pen(lays,D))
            loss.backward(); opt.step(); opt.zero_grad()
            if SMOKE and bi>=30: break
        P('pre-org epoch %d dense %.2f'%(e,acc(mb)))
        if METHOD=='spatial': net.swap(block=256)
    if METHOD=='spatial': D=tiling(mb,net)
    torch.save(mb.state_dict(),'%s/%s_preorg.pt'%(CKDIR,TAG))

masks=[torch.ones_like(wmat(l)) for l in lays]; out=[]

# ============================ MOVEMENT ============================
if METHOD=='movement':
    # block-movement: learnable score per block, straight-through top-v mask,
    # cubic ramp to each stage target during that stage's epochs
    S=[torch.zeros(Ro.shape[1],Co.shape[1],device=dev,requires_grad=True) for Ro,Co in D]
    prev=0.0
    for target in LEV:
        opt=torch.optim.AdamW([{'params':mb.parameters(),'lr':5e-5},{'params':S,'lr':1e-2}],weight_decay=0.05)
        steps=F_ep*len(trl); st=0
        for e in range(F_ep):
            mb.train()
            for bi,(x,y) in enumerate(trl):
                x,y=x.to(dev),y.to(dev)
                frac=prev+(target-prev)*(1-(1-min(1.,st/max(1,steps*0.8)))**3)
                allS=torch.cat([s.detach().flatten() for s in S])
                k=int(frac/100*allS.numel())
                thr=torch.kthvalue(allS,max(k,1)).values if k>0 else allS.min()-1
                orig_w=[l.weight.data.clone() for l in lays]
                for l,(Ro,Co),s in zip(lays,D,S):
                    hard=(Ro@(s>thr).float()@Co.t())
                    l.weight.data.mul_(emask(l,hard))          # hard mask fwd values
                loss=F.cross_entropy(mb(x),y)
                loss.backward()
                # movement update for scores: dL/dS_block = +sum_block(dL/dW * W)
                # (optimizer minimizes -> S accumulates -sum(g*w); grows where g*w<0)
                for l,(Ro,Co),s,w0 in zip(lays,D,S,orig_w):
                    if l.weight.grad is not None:
                        gs=Ro.t()@(gmat(l)*w0.view(wmat(l).shape))@Co
                        s.grad=gs if s.grad is None else s.grad+gs
                for l,w0 in zip(lays,orig_w): l.weight.data.copy_(w0)   # restore BEFORE step: dense weights train
                opt.step(); opt.zero_grad()
                st+=1
                if SMOKE and bi>=30: break
        prev=target
        allS=torch.cat([s.detach().flatten() for s in S]); k=int(target/100*allS.numel())
        thr=torch.kthvalue(allS,max(k,1)).values
        for (Ro,Co),s,msk in zip(D,S,masks): msk.copy_(1.-(Ro@(s<=thr).float()@Co.t()).clamp(0,1))
        for l,msk in zip(lays,masks):
            with torch.no_grad(): l.weight.mul_(emask(l,msk))
        finetune(mb,lays,masks,None,D,max(1,F_ep//3))          # brief stabilization at frozen mask
        a=acc(mb); bs_=blocksp(lays,D); out.append((bs_,a))
        P('tgt %d%% | blk-sp %.1f%% | acc %.1f'%(target,bs_,a))
        import pickle; pickle.dump(out,open('%s/%s.pkl'%(OUT,TAG),'wb'))
        torch.save(mb.state_dict(),'%s/%s_s%02d.pt'%(CKDIR,TAG,target))

# ============================ RIGL ============================
elif METHOD=='rigl':
    # per-target block-DST from the base (not iterative): drop |W|-block, grow |g|-block
    base_sd=copy.deepcopy(mb.state_dict())
    for target in LEV:
        mb.load_state_dict(base_sd)
        masks=[torch.ones_like(wmat(l)) for l in lays]
        # random init mask at target
        nb=[(Ro.shape[1],Co.shape[1]) for Ro,Co in D]
        allb=torch.cat([torch.zeros(a*b) for a,b in nb]); tot=allb.numel()
        keep=torch.zeros(tot,dtype=torch.bool); idx=torch.randperm(tot)[:int((1-target/100)*tot)]
        keep[idx]=True; o=0
        for (Ro,Co),msk in zip(D,masks):
            a,b=Ro.shape[1],Co.shape[1]
            kb=keep[o:o+a*b].reshape(a,b).float().to(dev); o+=a*b
            msk.copy_(Ro@kb@Co.t())
        for l,msk in zip(lays,masks):
            with torch.no_grad(): l.weight.mul_(emask(l,msk))
        opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
        steps=F_ep*len(trl); st=0; UPD=100
        for e in range(F_ep):
            mb.train()
            for bi,(x,y) in enumerate(trl):
                x,y=x.to(dev),y.to(dev)
                loss=F.cross_entropy(mb(x),y); loss.backward()
                if st%UPD==0 and st<steps*0.8:
                    fdrop=0.3*(1+np.cos(np.pi*st/(steps*0.8)))/2
                    for l,(Ro,Co),msk in zip(lays,D,masks):
                        wb=(Ro.t()@(wmat(l).detach().abs())@Co)      # active block mass
                        gb=(Ro.t()@(gmat(l).abs())@Co)               # grad mass (dense grads)
                        act=(Ro.t()@msk@Co)>0
                        na=int(act.sum()); k=int(fdrop*na)
                        if k<1 or na<=k: continue
                        wa=wb.clone(); wa[~act]=float('inf')
                        drop=torch.topk(-wa.flatten(),k).indices
                        gi=gb.clone(); gi[act]=-float('inf')
                        grow=torch.topk(gi.flatten(),k).indices
                        zb=torch.zeros_like(wb).flatten(); zb[drop]=1.
                        gz=torch.zeros_like(wb).flatten(); gz[grow]=1.
                        msk.mul_(1.-(Ro@zb.reshape(wb.shape)@Co.t())); msk.add_((Ro@gz.reshape(wb.shape)@Co.t())).clamp_(0,1)
                        with torch.no_grad(): l.weight.mul_(emask(l,msk))  # grown blocks start at 0
                opt.step(); opt.zero_grad()
                with torch.no_grad():
                    for l,msk in zip(lays,masks): l.weight.mul_(emask(l,msk))
                st+=1
                if SMOKE and bi>=30: break
        a=acc(mb); bs_=blocksp(lays,D); out.append((bs_,a))
        P('tgt %d%% | blk-sp %.1f%% | acc %.1f'%(target,bs_,a))
        import pickle; pickle.dump(out,open('%s/%s.pkl'%(OUT,TAG),'wb'))
        torch.save(mb.state_dict(),'%s/%s_s%02d.pt'%(CKDIR,TAG,target))

# ============================ ITERATIVE (mag/taylor/spatial/glasso) ============================
else:
    for target in LEV:
        if METHOD=='taylor' or VARIANT=='taylorscore':
            scores=taylor_scores(mb,lays,D,usenet)
        else:
            scores=mag_scores(lays,D,masks)
        scores=layer_normalize(scores)
        prune_to(target,lays,D,masks,scores)
        P('tgt %d%% | post-cut acc %.2f'%(target,acc(mb)))
        finetune(mb,lays,masks,usenet,D,F_ep,gl=(METHOD=='glasso'))
        a=acc(mb); bs_=blocksp(lays,D); out.append((bs_,a))
        P('tgt %d%% | blk-sp %.1f%% | acc %.1f'%(target,bs_,a))
        import pickle; pickle.dump(out,open('%s/%s.pkl'%(OUT,TAG),'wb'))
        torch.save(mb.state_dict(),'%s/%s_s%02d.pt'%(CKDIR,TAG,target))
print('done',flush=True)
