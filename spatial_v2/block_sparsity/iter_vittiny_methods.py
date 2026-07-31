import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import timm
METHOD=sys.argv[1]                       # l1 | spatial0   (block15 already done separately)
STR=[float(g) for g in sys.argv[2].split(',')]
dev='cuda'; torch.manual_seed(0); np.random.seed(0)
Xtr,Ytr=torch.load('/home/judah/spatial_v2/block_sparsity/vits_cache_train.pt'); Xtr=Xtr.to(dev); Ytr=Ytr.to(dev)
Xte,Yte=torch.load('/home/judah/spatial_v2/block_sparsity/vits_cache_test.pt'); Xte=Xte.to(dev); Yte=Yte.to(dev)
def norm(x): return x.float().div_(127.5).sub_(1.)
def train_batches(bs=64):
    perm=torch.randperm(Xtr.shape[0],device=dev)
    for i in range(0,len(perm),bs):
        ix=perm[i:i+bs]; x=norm(Xtr[ix])
        if torch.rand(1).item()<0.5: x=torch.flip(x,[3])
        yield x,Ytr[ix]
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def l1m(m):
    t=0.;n=0
    for l in regl(m): t=t+l.weight.abs().sum(); n+=l.weight.numel()
    return t/n
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for i in range(0,Xte.shape[0],256):
        x=norm(Xte[i:i+256]); y=Yte[i:i+256]; c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100.*c/t
LEV=[0,10,20,30,40,50,60,70,80,85,90,95]
t0=time.time(); P=lambda s: print('[%6.1fs] %s'%(time.time()-t0,s),flush=True)
res={}
for S in STR:
    torch.manual_seed(0); np.random.seed(0)
    base=timm.create_model('vit_tiny_patch16_224',pretrained=True,num_classes=100).to(dev)
    if METHOD=='spatial0':
        net=swc.SpatialCNN(base,gamma=S,device=dev,block_size=0).to(dev); m=net.model
    else:
        m=base; net=None
    opt=torch.optim.AdamW([{'params':m.parameters(),'lr':1e-4}],weight_decay=0.05)
    P('%s=%g start'%(METHOD,S))
    for e in range(4):
        m.train()
        if net is not None: net.swap(block=256)
        for x,y in train_batches(64):
            loss=F.cross_entropy((net(x) if net is not None else m(x)),y)
            loss=loss+(net.get_cost() if net is not None else S*l1m(m))
            loss.backward(); opt.step(); opt.zero_grad()
        P('%s=%g ep%d done'%(METHOD,S,e))
    lays=regl(m); base_acc=acc(m)
    with torch.no_grad():
        orig=[l.weight.detach().clone() for l in lays]
        allw=torch.cat([wmat(l).abs().flatten() for l in lays])
        curve=[]
        for p in LEV:
            for l,w0 in zip(lays,orig): l.weight.copy_(w0)
            if p>0:
                k=max(1,int(p/100.*allw.numel())); thr=torch.kthvalue(allw,k).values
                for l in lays:
                    w=wmat(l); w.mul_((w.abs()>thr).float())
            curve.append((p,acc(m)))
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
    res[S]={'base':base_acc,'curve':curve}
    P('%s=%g DONE base %.1f | '%(METHOD,S,base_acc)+' '.join('%d:%.1f'%(p,a) for p,a in curve))
    import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/vittiny_%s_%s.pkl'%(METHOD,'_'.join('%g'%g for g in STR)),'wb'))
print('done',flush=True)
