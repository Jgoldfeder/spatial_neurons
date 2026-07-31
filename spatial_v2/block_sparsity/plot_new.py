import pickle,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SP=[0,10,20,30,40,50,60,70,80,85,90,95]; S=np.array(SP,float)
SPEC=[('l1','L1 (unstructured)','#E69F00'),
      ('swap','spatial-block swap B=15 (frozen)','#0072B2'),
      ('swaplearn','spatial-block swap B=15 + learned pos','#009E73'),
      ('swap_b1','spatial swap B=1 (per-neuron)','#CC79A7')]
def load(meth):
    try: return pickle.load(open('/home/judah/spatial_v2/block_sparsity/new_models/%s_stats.pkl'%meth,'rb'))['stats']
    except: return {}
def mono(curves):
    best=np.full(len(S),-1.0)
    for c in curves: best=np.maximum(best,np.array([dict(c)[s] for s in SP],float))
    for i in range(len(best)-2,-1,-1): best[i]=max(best[i],best[i+1])
    return best
fig,ax=plt.subplots(1,2,figsize=(14,5.8))
for meth,name,col in SPEC:
    st=load(meth)
    if not st: continue
    ft=sorted((v['natsp'],v['ft_acc']) for v in st.values())
    ax[0].plot([p[0] for p in ft],[p[1] for p in ft],'-o',color=col,lw=2,ms=9,label='%s (n=%d)'%(name,len(st)))
    ax[1].plot(S,mono([v['curve'] for v in st.values()]),'o-',color=col,lw=2.8,ms=5,label=name)
ax[0].set_title('FIXED THRESHOLD (τ=1e-3): one point per model',fontsize=11)
ax[0].set_xlabel('natural sparsity at τ=1e-3 (%)'); ax[0].set_ylabel('pruned accuracy (%)'); ax[0].set_ylim(0,90); ax[0].set_xlim(0,100)
ax[1].set_title('ENVELOPE (best acc over strength, monotone)',fontsize=11)
ax[1].set_xlabel('unstructured weight sparsity (%)'); ax[1].set_ylabel('best acc at sparsity ≥ s (%)'); ax[1].set_ylim(0,90); ax[1].set_xlim(0,95)
for a in ax: a.grid(alpha=.3); a.legend(fontsize=9,loc='lower left')
fig.suptitle('vit_tiny / CIFAR-100, 4-epoch (new_models) — L1 vs spatial-block swap vs swap+learned pos',fontsize=11.5)
fig.tight_layout(rect=[0,0,1,.95])
out='/home/judah/spatial_v2/block_sparsity/vittiny_new_2view.png'; fig.savefig(out,dpi=130); print('saved',out)
for meth,name,_ in SPEC:
    st=load(meth)
    print('%-30s'%name,[(round(v['natsp'],0),round(v['ft_acc'],0)) for v in sorted(st.values(),key=lambda z:z['natsp'])] if st else '-')
