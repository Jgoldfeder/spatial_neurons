import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import timm
GAMMAS=[float(g) for g in sys.argv[1].split(',')]
dev='cuda'; B=15; torch.manual_seed(0); np.random.seed(0)
# cached 224 uint8 tensors -> normalize+flip on GPU, zero CPU workers (immune to resize-pipeline starvation)
Xtr,Ytr=torch.load('/home/judah/spatial_v2/block_sparsity/vits_cache_train.pt'); Xtr=Xtr.to(dev); Ytr=Ytr.to(dev)
Xte,Yte=torch.load('/home/judah/spatial_v2/block_sparsity/vits_cache_test.pt'); Xte=Xte.to(dev); Yte=Yte.to(dev)
def norm(x): return x.float().div_(127.5).sub_(1.)     # matches Normalize((.5,)*3,(.5,)*3)
def train_batches(bs=64):
    perm=torch.randperm(Xtr.shape[0],device=dev)
    for i in range(0,len(perm),bs):
        ix=perm[i:i+bs]; x=norm(Xtr[ix]);
        if torch.rand(1).item()<0.5: x=torch.flip(x,[3])   # cheap GPU hflip (batch-level)
        yield x,Ytr[ix]
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.tensor(ids,device=dev)]=1.; return O
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for i in range(0,Xte.shape[0],256):
        x=norm(Xte[i:i+256]); y=Yte[i:i+256]; c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
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
P=lambda s: print('[%6.1fs] %s'%(time.time()-t0,s),flush=True)
res={}
for GAMMA in GAMMAS:
    torch.manual_seed(0); np.random.seed(0)
    base=timm.create_model('vit_small_patch16_224',pretrained=True,num_classes=100).to(dev)
    net=swc.SpatialCNN(base,gamma=GAMMA,device=dev,block_size=B).to(dev); m=net.model
    opt=torch.optim.AdamW([{'params':m.parameters(),'lr':1e-4}],weight_decay=0.05)
    P('g=%g start'%GAMMA)
    for e in range(4):
        m.train(); ts=time.time(); net.swap(block=256); P('g=%g ep%d swap %.0fs'%(GAMMA,e,time.time()-ts))
        tt=time.time()
        for x,y in train_batches(64):
            (F.cross_entropy(net(x),y)+net.get_cost()).backward(); opt.step(); opt.zero_grad()
        P('g=%g ep%d train %.0fs'%(GAMMA,e,time.time()-tt))
    lays=regl(m)
    base_acc=acc(m)
    with torch.no_grad():
        orig=[l.weight.detach().clone() for l in lays]
        allw=torch.cat([wmat(l).abs().flatten() for l in lays])   # global unstructured magnitude prune
        curve=[]
        for p in LEV:
            for l,w0 in zip(lays,orig): l.weight.copy_(w0)
            if p>0:
                k=max(1,int(p/100.*allw.numel())); thr=torch.kthvalue(allw,k).values
                for l in lays:
                    w=wmat(l); w.mul_((w.abs()>thr).float())
            curve.append((p,acc(m)))
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
    res[GAMMA]={'base':base_acc,'curve':curve}
    P('g=%g DONE base %.1f | '%(GAMMA,base_acc)+' '.join('%d:%.1f'%(p,a) for p,a in curve))
    print('[vits15c g=%g] base %.1f | '%(GAMMA,base_acc)+' '.join('%d:%.1f'%(p,a) for p,a in curve)+' (%.0fs)'%(time.time()-t0),flush=True)
    import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/vitsmall_b15_%s.pkl'%('_'.join('%g'%g for g in GAMMAS)),'wb'))
print('done',flush=True)
