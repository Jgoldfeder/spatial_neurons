import sys,os,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import timm
METHOD=sys.argv[1]                       # l1 | swap | swaplearn
STR=[float(g) for g in sys.argv[2].split(',')]
B=int(sys.argv[3]) if len(sys.argv)>3 else 15
TAG=METHOD if (METHOD=='l1' or B==15) else '%s_b%d'%(METHOD,B)   # distinct outputs when block size != 15
dev='cuda'; torch.manual_seed(0); np.random.seed(0)
MDIR='/home/judah/spatial_v2/block_sparsity/new_models'; os.makedirs(MDIR,exist_ok=True)
SPARS=[0,10,20,30,40,50,60,70,80,85,90,95]; TAU=1e-3
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
def unique_sites(x,y):
    k=np.round(np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1),5)
    uniq,inv=np.unique(k,axis=0,return_inverse=True)
    return torch.tensor(uniq,dtype=torch.float32,device=dev), torch.tensor(inv,dtype=torch.long,device=dev)
def recover(x,y,SP): return torch.cdist(torch.stack([x,y],1),SP).argmin(1)
t0=time.time(); P=lambda s: print('[%6.1fs] %s'%(time.time()-t0,s),flush=True)
import pickle
try: stats=pickle.load(open('%s/%s_stats.pkl'%(MDIR,TAG),'rb'))['stats']; P('loaded %d existing'%len(stats))
except: stats={}
for S in STR:
    torch.manual_seed(0); np.random.seed(0)
    base=timm.create_model('vit_tiny_patch16_224',pretrained=True,num_classes=100).to(dev)
    net=None; SP=None; asg=None
    if METHOD=='l1':
        m=base; opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.05)
    else:
        net=swc.SpatialCNN(base,gamma=S,device=dev,block_size=B).to(dev); m=net.model; net.swap(block=256)
        if METHOD=='swaplearn':
            L=len(net.layers); SP=[]; asg=[]
            for l in range(L):
                xi,yi,xo,yo=net.planes(l); si,ai=unique_sites(xi,yi); so,ao=unique_sites(xo,yo)
                SP.append([nn.Parameter(si),nn.Parameter(so)]); asg.append([ai,ao])
            spp=[p for pr in SP for p in pr]
            opt=torch.optim.AdamW([{'params':m.parameters(),'lr':1e-4},{'params':spp,'lr':1e-2}],weight_decay=0.05)
        else:
            opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.05)
    def wiring_cost():
        wire=0.; nparam=0
        for l in range(len(net.layers)):
            W=net.weight_matrix(l).abs(); (SPi,SPo)=SP[l]; (ai,ao)=asg[l]
            xin=SPi[ai,0]; yin=SPi[ai,1]; xout=SPo[ao,0]; yout=SPo[ao,1]
            dx=xout[:,None]-xin[None,:]; dy=yout[:,None]-yin[None,:]
            dist=torch.sqrt(dx*dx+dy*dy+net.D**2); wire=wire+(W*dist).sum(); nparam+=W.numel()
        return S*(wire/nparam)
    P('%s=%g start'%(METHOD,S))
    for e in range(4):
        m.train()
        if METHOD=='swap': net.swap(block=256)
        if METHOD=='swaplearn':
            with torch.no_grad():
                for l in range(len(net.layers)):
                    xi,yi,xo,yo=net.planes(l); (SPi,SPo)=SP[l]; (ai,ao)=asg[l]
                    xi.copy_(SPi[ai,0]); yi.copy_(SPi[ai,1]); xo.copy_(SPo[ao,0]); yo.copy_(SPo[ao,1])
            net.swap(block=256)
            for l in range(len(net.layers)):
                xi,yi,xo,yo=net.planes(l); (SPi,SPo)=SP[l]; asg[l]=[recover(xi,yi,SPi),recover(xo,yo,SPo)]
        for x,y in train_batches(64):
            loss=F.cross_entropy((net(x) if net is not None else m(x)),y)
            loss=loss+(S*l1m(m) if METHOD=='l1' else (wiring_cost() if METHOD=='swaplearn' else net.get_cost()))
            loss.backward(); opt.step(); opt.zero_grad()
    # ---- stats at end ----
    lays=regl(m); base_acc=acc(m)
    with torch.no_grad():
        orig=[l.weight.detach().clone() for l in lays]
        allw=torch.cat([wmat(l).abs().flatten() for l in lays])
        natsp=100.*(allw<TAU).float().mean().item()
        # fixed-threshold pruned accuracy (remove |w|<tau)
        for l in lays: w=wmat(l); w.mul_((w.abs()>=TAU).float())
        ft_acc=acc(m)
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
        # magnitude prune curve
        curve=[]
        for s in SPARS:
            for l,w0 in zip(lays,orig): l.weight.copy_(w0)
            if s>0:
                k=max(1,int(s/100.*allw.numel())); thr=torch.kthvalue(allw,k).values
                for l in lays: w=wmat(l); w.mul_((w.abs()>thr).float())
            curve.append((s,acc(m)))
        for l,w0 in zip(lays,orig): l.weight.copy_(w0)
    # ---- save model + positions ----
    torch.save(m.state_dict(), '%s/%s_%g.pt'%(MDIR,TAG,S))
    if net is not None: torch.save({'pos':[p.detach().cpu() for p in net.pos]}, '%s/%s_%g_pos.pt'%(MDIR,TAG,S))
    stats[S]={'base':base_acc,'natsp':natsp,'ft_acc':ft_acc,'curve':curve}
    P('%s=%g DONE base %.1f natsp %.1f ft %.1f | '%(METHOD,S,base_acc,natsp,ft_acc)+' '.join('%d:%.1f'%(s,a) for s,a in curve))
    import pickle; pickle.dump({'method':METHOD,'B':B,'SPARS':SPARS,'stats':stats},open('%s/%s_stats.pkl'%(MDIR,TAG),'wb'))
print('done',flush=True)
