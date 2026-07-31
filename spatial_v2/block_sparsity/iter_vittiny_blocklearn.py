import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import timm
GAMMAS=[float(g) for g in sys.argv[1].split(',')]
dev='cuda'; B=15; torch.manual_seed(0); np.random.seed(0)
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
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for i in range(0,Xte.shape[0],256):
        x=norm(Xte[i:i+256]); y=Yte[i:i+256]; c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100.*c/t
def unique_sites(x,y):
    k=np.round(np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1),5)
    uniq,inv=np.unique(k,axis=0,return_inverse=True)
    return torch.tensor(uniq,dtype=torch.float32,device=dev), torch.tensor(inv,dtype=torch.long,device=dev)
def recover(x,y,SP):                       # per-neuron (x,y) -> nearest site index in SP (nsites,2)
    P=torch.stack([x,y],1); return torch.cdist(P,SP).argmin(1)
LEV=[0,10,20,30,40,50,60,70,80,85,90,95]
t0=time.time(); P=lambda s: print('[%6.1fs] %s'%(time.time()-t0,s),flush=True)
res={}
for GAMMA in GAMMAS:
    torch.manual_seed(0); np.random.seed(0)
    base=timm.create_model('vit_tiny_patch16_224',pretrained=True,num_classes=100).to(dev)
    net=swc.SpatialCNN(base,gamma=GAMMA,device=dev,block_size=B).to(dev); m=net.model
    L=len(net.layers)
    net.swap(block=256)                    # initial block assignment
    # learnable SITE coords per layer/plane + neuron->site assignment
    SP=[]; asg=[]
    for l in range(L):
        xi,yi,xo,yo=net.planes(l)
        si,ai=unique_sites(xi,yi); so,ao=unique_sites(xo,yo)
        SP.append([nn.Parameter(si),nn.Parameter(so)]); asg.append([ai,ao])
    sp_params=[p for pair in SP for p in pair]
    opt=torch.optim.AdamW([{'params':m.parameters(),'lr':1e-4},{'params':sp_params,'lr':1e-2}],weight_decay=0.05)
    def wiring_cost():
        wire=0.; nparam=0
        for l in range(L):
            W=net.weight_matrix(l).abs(); (SPi,SPo)=SP[l]; (ai,ao)=asg[l]
            xin=SPi[ai,0]; yin=SPi[ai,1]; xout=SPo[ao,0]; yout=SPo[ao,1]
            dx=xout[:,None]-xin[None,:]; dy=yout[:,None]-yin[None,:]
            dist=torch.sqrt(dx*dx+dy*dy+net.D**2); wire=wire+(W*dist).sum(); nparam+=W.numel()
        return GAMMA*(wire/nparam)
    P('g=%g start (%d layers, learnable block positions)'%(GAMMA,L))
    for e in range(4):
        m.train(); tt=time.time()
        for x,y in train_batches(64):
            (F.cross_entropy(net(x),y)+wiring_cost()).backward(); opt.step(); opt.zero_grad()
        # re-swap: sync learned site coords into net.pos, reassign neurons, recover asg
        with torch.no_grad():
            for l in range(L):
                xi,yi,xo,yo=net.planes(l); (SPi,SPo)=SP[l]; (ai,ao)=asg[l]
                xi.copy_(SPi[ai,0]); yi.copy_(SPi[ai,1]); xo.copy_(SPo[ao,0]); yo.copy_(SPo[ao,1])
        net.swap(block=256)
        for l in range(L):
            xi,yi,xo,yo=net.planes(l); (SPi,SPo)=SP[l]
            asg[l]=[recover(xi,yi,SPi), recover(xo,yo,SPo)]
        P('g=%g ep%d done (%.0fs)'%(GAMMA,e,time.time()-tt))
    lays=regl(m); base_acc=acc(m)
    with torch.no_grad():
        orig=[l.weight.detach().clone() for l in lays]
        allw=torch.cat([wmat(l).abs().flatten() for l in lays])
        natsp=100.*(allw<1e-3).float().mean().item()              # natural sparsity at tau=1e-3
        curve=[]
        for p in LEV:
            for l,w0 in zip(lays,orig): l.weight.copy_(w0)
            if p>0:
                k=max(1,int(p/100.*allw.numel())); thr=torch.kthvalue(allw,k).values
                for l in lays:
                    w=wmat(l); w.mul_((w.abs()>thr).float())
            curve.append((p,acc(m)))
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
    res[GAMMA]={'base':base_acc,'natsp':natsp,'curve':curve}
    P('g=%g DONE base %.1f natsp %.1f | '%(GAMMA,base_acc,natsp)+' '.join('%d:%.1f'%(p,a) for p,a in curve))
    import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/vittiny_blocklearn_%s.pkl'%('_'.join('%g'%g for g in GAMMAS)),'wb'))
print('done',flush=True)
