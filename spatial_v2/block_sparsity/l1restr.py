import pickle,glob,numpy as np,torch,torch.nn as nn,torchvision,warnings
warnings.filterwarnings('ignore')
B=16; SPARS=[0,10,20,30,40,50,60,70,80,85,90,95]
r={}
for f in glob.glob('rn_regime2_shard*.pkl'):
    try: r.update(pickle.load(open(f,'rb'))['rec'])
    except: pass
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def reorder(M,iters=5):
    o,i=M.shape; ro=np.arange(o); co=np.arange(i); ai=np.arange(i); ao=np.arange(o)[:,None]
    for _ in range(iters):
        C=M[ro][:,co]; ro=ro[np.argsort((C*ai).sum(1)/(C.sum(1)+1e-9),kind='stable')]
        C=M[ro][:,co]; co=co[np.argsort((C*ao).sum(0)/(C.sum(0)+1e-9),kind='stable')]
    return ro,co
def empty_frac(layers,thr):
    num=den=0
    for l in layers:
        W=wmat(l).abs().detach().numpy(); o,i=W.shape
        if o<B or i<B: continue
        alive=W>=thr; ro,co=reorder(alive.astype(float)); alive=alive[ro][:,co]
        no,ni=o//B,i//B; blk=alive[:no*B,:ni*B].reshape(no,B,ni,B).any(axis=(1,3))
        num+=int((~blk).sum()); den+=blk.size
    return 100.*num/max(den,1)
def load(g,e):
    m=torchvision.models.resnet18(); m.fc=nn.Linear(512,100)
    m.load_state_dict(torch.load('rn_regime2_models/l1_%g_ep%02d.pt'%(g,e),map_location='cpu')); return m
P=[]
for g in [1000,1500,2000]:
    for e in sorted(e for (mo,gg,e) in r if mo=='l1' and gg==g)[-3:]:
        m=load(g,e); lays=regl(m)
        allw=np.concatenate([wmat(l).abs().flatten().detach().numpy() for l in lays])
        q=r[('l1',g,e)]['q']
        for j,s in enumerate(SPARS):
            thr=float(np.quantile(allw,s/100.)) if s>0 else 1e-12
            P.append((q[j],empty_frac(lays,thr)))
def mx(fl): v=[bs for a,bs in P if a>=fl]; return max(v) if v else 0.0
print('L1 restricted (g=1000,1500,2000; base 70-73) max empty-block %% at acc floor:')
for fl in [72,70,68,65,60,55,50]: print('  acc>=%d%%: %.1f%%'%(fl,mx(fl)))
pickle.dump(P,open('/home/judah/spatial_v2/block_sparsity/l1restr.pkl','wb')); print('done')
