import pickle,glob,numpy as np,torch,torch.nn as nn,torchvision,warnings
warnings.filterwarnings('ignore'); torch.set_num_threads(4)
B=16; SPARS=[0,10,20,30,40,50,60,70,80,85,90,95]
rr={}
for f in glob.glob('rn_regime2_shard*.pkl'):
    try: rr.update(pickle.load(open(f,'rb'))['rec'])
    except: pass
br={}
for f in glob.glob('blk_shard*.pkl'):
    try: br.update(pickle.load(open(f,'rb'))['rec'])
    except: pass
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def reorder(M,iters=6):
    o,i=M.shape; ro=np.arange(o); co=np.arange(i); ai=np.arange(i); ao=np.arange(o)[:,None]
    for _ in range(iters):
        C=M[ro][:,co]; ro=ro[np.argsort((C*ai).sum(1)/(C.sum(1)+1e-9),kind='stable')]
        C=M[ro][:,co]; co=co[np.argsort((C*ao).sum(0)/(C.sum(0)+1e-9),kind='stable')]
    return ro,co
def ef(lays,thr,rw):
    num=den=0
    for l in lays:
        W=wmat(l).abs().detach().numpy(); o,i=W.shape
        if o<B or i<B: continue
        a=W>=thr
        if rw: ro,co=reorder(a.astype(float)); a=a[ro][:,co]
        no,ni=o//B,i//B; blk=a[:no*B,:ni*B].reshape(no,B,ni,B).any(axis=(1,3))
        num+=int((~blk).sum()); den+=blk.size
    return 100.*num/max(den,1)
# --- swap g=256 final epoch, trained-tiling blk straight from stored qb ---
e=max(ee for (mo,g,ee) in br if mo=='swap' and g==256); r=br[('swap',256.0,e)]
print('SWAP g=256 ep%d (trained tiling):'%e)
print(' sp%%  acc   blk(trained)')
for j,s in enumerate(SPARS): print(' %2d%%  %5.1f   %5.1f'%(s,r['qa'][j],r['qb'][j]))
# --- L1 g=1000 & 1500 final epoch: acc from stored q, blk with & without reorder ---
for g in [1000,1500]:
    e=max(ee for (mo,gg,ee) in rr if mo=='l1' and gg==g)
    m=torchvision.models.resnet18(); m.fc=nn.Linear(512,100)
    m.load_state_dict(torch.load('rn_regime2_models/l1_%g_ep%02d.pt'%(g,e),map_location='cpu')); lays=regl(m)
    allw=np.concatenate([wmat(l).abs().flatten().detach().numpy() for l in lays]); q=rr[('l1',float(g),e)]['q']
    print('\nL1 g=%d ep%d:  sp%%  acc   blk(reorder)  blk(NOreorder)'%(g,e))
    for j,s in enumerate(SPARS):
        thr=float(np.quantile(allw,s/100.)) if s>0 else 1e-12
        print('       %2d%%  %5.1f   %6.1f        %6.1f'%(s,q[j],ef(lays,thr,True),ef(lays,thr,False)))
print('done')
