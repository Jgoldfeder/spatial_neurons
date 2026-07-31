import torch,warnings,pickle
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
dev='cuda'; dt=torch.float16; M=K=4096; N=4096; torch.manual_seed(0)
def ct(fn,warm=40,it=150):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); ts=[]
    for _ in range(it): s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts)//2]
def bytes_of(t):
    b=t.values().numel()*t.values().element_size()
    for a in ['col_indices','crow_indices']:
        if hasattr(t,a): b+=getattr(t,a)().numel()*getattr(t,a)().element_size()
    return b
x=torch.randn(K,N,device=dev,dtype=dt); Wd=torch.randn(M,K,device=dev,dtype=dt)
dms=ct(lambda:Wd@x); dmem=Wd.numel()*Wd.element_size()
gpu=torch.cuda.get_device_name(0)
print('DENSE %.3f ms | %.1f TFLOPS | %s'%(dms,2*M*K*N/dms/1e9,gpu),flush=True)
BLOCKS=[16,32,64,128]
SP=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
res={'dense':{'ms':dms,'mem':dmem},'gpu':gpu,'bsr':{},'csr':{}}
for B in BLOCKS:
    for p in SP:
        W=torch.randn(M,K,device=dev,dtype=dt); Wb=W.view(M//B,B,K//B,B)
        mask=(torch.rand(M//B,K//B,device=dev)>=p); W=(Wb*mask[:,None,:,None]).reshape(M,K).contiguous()
        act=float((~mask).float().mean()); bsr=W.to_sparse_bsr(B)
        res['bsr'][(B,round(act,4))]={'ms':ct(lambda:bsr@x),'mem':bytes_of(bsr),'sp':act}
    print('B=%d done'%B,flush=True)
for p in SP:
    W=torch.randn(M,K,device=dev,dtype=dt); W=(W*(torch.rand(M,K,device=dev)>=p)).contiguous()
    csr=W.to_sparse_csr(); res['csr'][round(p,4)]={'ms':ct(lambda:torch.sparse.mm(csr,x)),'mem':bytes_of(csr),'sp':p}
print('csr done',flush=True)
pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/gpubench.pkl','wb'))
# ---- plot ----
import numpy as np
cols={16:'#f4a3a3',32:'tab:red',64:'darkred',128:'purple'}
fig,(a,b)=plt.subplots(1,2,figsize=(16,6.5))
for B in BLOCKS:
    ks=sorted([k for k in res['bsr'] if k[0]==B],key=lambda k:k[1])
    sp=np.array([res['bsr'][k]['sp'] for k in ks]); spd=np.array([dms/res['bsr'][k]['ms'] for k in ks])
    a.plot(sp*100,spd,'-o',color=cols[B],label='BSR %dx%d'%(B,B),lw=2,ms=4)
    be=np.interp(1.0,spd,sp)*100
    a.annotate('%.0f%%'%be,(be,1.0),textcoords='offset points',xytext=(0,-14),fontsize=8,color=cols[B],ha='center')
cs=sorted(res['csr']); a.plot([res['csr'][p]['sp']*100 for p in cs],[dms/res['csr'][p]['ms'] for p in cs],'--s',color='tab:blue',label='unstructured (cuSPARSE CSR)',lw=2,ms=4)
a.axhline(1,color='k',lw=1,ls=':'); a.set_yscale('log'); a.set_xlabel('sparsity (%)'); a.set_ylabel('speedup vs dense (log)')
a.set_title('GPU wall-time speedup — fp16 4096^3 on %s'%gpu.replace('NVIDIA GeForce ','')); a.legend(fontsize=9); a.grid(alpha=.3,which='both')
for B in BLOCKS:
    ks=sorted([k for k in res['bsr'] if k[0]==B],key=lambda k:k[1])
    b.plot([res['bsr'][k]['sp']*100 for k in ks],[res['bsr'][k]['mem']/dmem for k in ks],'-o',color=cols[B],label='BSR %dx%d'%(B,B),lw=2,ms=4)
b.plot([res['csr'][p]['sp']*100 for p in cs],[res['csr'][p]['mem']/dmem for p in cs],'--s',color='tab:blue',label='CSR int64',lw=2,ms=4)
b.axhline(1,color='k',lw=1,ls=':'); b.set_xlabel('sparsity (%)'); b.set_ylabel('storage memory / dense'); b.set_title('Storage memory'); b.legend(fontsize=9); b.grid(alpha=.3); b.set_ylim(0,2)
plt.tight_layout(); plt.savefig('/home/judah/spatial_v2/block_sparsity/gpubench.png',dpi=125,bbox_inches='tight')
print('break-evens:',{B:round(float(np.interp(1.0,np.array([dms/res['bsr'][k]['ms'] for k in sorted([k for k in res['bsr'] if k[0]==B],key=lambda k:k[1])]),np.array([res['bsr'][k]['sp'] for k in sorted([k for k in res['bsr'] if k[0]==B],key=lambda k:k[1])]))*100),1) for B in BLOCKS})
print('saved gpubench.png + gpubench.pkl')
print('done')
