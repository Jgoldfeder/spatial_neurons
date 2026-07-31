import pickle,glob,numpy as np,torch,torch.nn as nn,torchvision,warnings
warnings.filterwarnings('ignore')
B=16; SPARS=[0,10,20,30,40,50,60,70,80,85,90,95]
br={}
for f in glob.glob('blk_shard*.pkl'):
    try: br.update(pickle.load(open(f,'rb'))['rec'])
    except: pass
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def reorder(M,iters=5):
    o,i=M.shape; ro=np.arange(o); co=np.arange(i); ai=np.arange(i); ao=np.arange(o)[:,None]
    for _ in range(iters):
        C=M[ro][:,co]; ro=ro[np.argsort((C*ai).sum(1)/(C.sum(1)+1e-9),kind='stable')]
        C=M[ro][:,co]; co=co[np.argsort((C*ao).sum(0)/(C.sum(0)+1e-9),kind='stable')]
    return ro,co
def empty_frac(layers,thr,do_reorder):
    num=den=0
    for l in layers:
        W=wmat(l).abs().detach().numpy(); o,i=W.shape
        if o<B or i<B: continue
        alive=W>=thr
        if do_reorder: ro,co=reorder(alive.astype(float)); alive=alive[ro][:,co]
        no,ni=o//B,i//B; blk=alive[:no*B,:ni*B].reshape(no,B,ni,B).any(axis=(1,3))
        num+=int((~blk).sum()); den+=blk.size
    return 100.*num/max(den,1)
def load(mode,g,e):
    m=torchvision.models.resnet18(); m.fc=nn.Linear(512,100)
    m.load_state_dict(torch.load('blk_models/%s_%g_ep%02d.pt'%(mode,g,e),map_location='cpu')); return m
def pts(mode,gs,neps,do_reorder):
    P=[]
    for g in gs:
        eps=sorted(e for (mo,gg,e) in br if mo==mode and gg==g)[-neps:]
        for e in eps:
            m=load(mode,g,e); lays=regl(m)
            allw=np.concatenate([wmat(l).abs().flatten().detach().numpy() for l in lays])
            qa=br[(mode,float(g),e)]['qa']
            for j,s in enumerate(SPARS):
                thr=float(np.quantile(allw,s/100.)) if s>0 else 1e-12
                P.append((qa[j],empty_frac(lays,thr,do_reorder)))
    return P
G=[64,128,256]
swR=pts('swap',G,3,True); nsR=pts('noswap',G,3,True)
# no-reorder trained-tiling numbers straight from stored qb
def bpts(mode):
    P=[]
    for (mo,gg,e),rr in br.items():
        if mo==mode:
            for j in range(len(SPARS)): P.append((rr['qa'][j],rr['qb'][j]))
    return P
sw0,ns0=bpts('swap'),bpts('noswap')
def mx(P,fl): v=[bs for a,bs in P if a>=fl]; return max(v) if v else 0.0
print('max empty-16x16-block %% at accuracy floor (L1+reorder=34.1 ref):')
print('%6s | %10s %11s | %10s %12s'%('acc>=','swap trained','swap+REORDER','noswap trnd','noswap+REORDER'))
for fl in [72,70,68,65,60,55,50]:
    print('%5d%% | %9.1f%% %11.1f%% | %9.1f%% %13.1f%%'%(fl,mx(sw0,fl),mx(swR,fl),mx(ns0,fl),mx(nsR,fl)))
pickle.dump({'swR':swR,'nsR':nsR},open('/home/judah/spatial_v2/block_sparsity/spatial_reorder.pkl','wb')); print('done')
