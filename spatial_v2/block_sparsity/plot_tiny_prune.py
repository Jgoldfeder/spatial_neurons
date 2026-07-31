import pickle,glob,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SP=[0,10,20,30,40,50,60,70,80,85,90,95]; S=np.array(SP,float)

# ---- regime2 (L1 + spatial non-block): thr=(natsp,acc) fixed-threshold; q=mag-prune curve ----
R2=[pickle.load(open(f,'rb'))['rec'] for f in sorted(glob.glob('vittiny_regime2_shard*.pkl'))]
def r2(meth):
    hps=sorted(set(k[1] for r in R2 for k in r if k[0]==meth))
    ftpts=[]; curves=[]
    for hp in hps:
        thrs=[]; qs=[]
        for r in R2:
            es=[k[2] for k in r if k[0]==meth and k[1]==hp]
            if not es: continue
            e=max(es); thrs.append(r[(meth,hp,e)]['thr']); qs.append(np.array(r[(meth,hp,e)]['q'],float))
        if not qs: continue
        thr=np.mean(thrs,0); ftpts.append((thr[0],thr[1])); curves.append(np.mean(qs,0))
    return ftpts,curves

# ---- blk_shard (spatial-BLOCK, B=16): natsp + qa (mag-prune acc curve) ----
BK=[pickle.load(open(f,'rb'))['rec'] for f in sorted(glob.glob('blk_shard*.pkl'))]
def blk(meth='swap'):
    hps=sorted(set(k[1] for r in BK for k in r if k[0]==meth and k[1]>0))
    ftpts=[]; curves=[]
    for hp in hps:
        ns=[]; qs=[]
        for r in BK:
            es=[k[2] for k in r if k[0]==meth and k[1]==hp]
            if not es: continue
            e=max(es); ns.append(r[(meth,hp,e)]['natsp']); qs.append(np.array(r[(meth,hp,e)]['qa'],float))
        if not qs: continue
        nsm=float(np.mean(ns)); qm=np.mean(qs,0)
        ftpts.append((nsm,float(np.interp(nsm,S,qm)))); curves.append(qm)
    return ftpts,curves

l1_ft,l1_c=r2('l1'); sp_ft,sp_c=r2('swap'); bk_ft,bk_c=blk('swap')
DATA=[('L1 (unstructured)','#E69F00',l1_ft,l1_c),
      ('spatial (swap, non-block)','#0072B2',sp_ft,sp_c),
      ('spatial-BLOCK (swap, B=16)','#009E73',bk_ft,bk_c)]

def mono_env(curves):
    best=np.full(len(S),-1.0)
    for q in curves: best=np.maximum(best,q)
    for i in range(len(best)-2,-1,-1): best[i]=max(best[i],best[i+1])
    return best

fig,ax=plt.subplots(1,2,figsize=(14,5.8))
for name,col,ft,cur in DATA:
    ft=sorted(ft); X=[p[0] for p in ft]; Y=[p[1] for p in ft]
    ax[0].plot(X,Y,'-o',color=col,lw=2,ms=9,label=name)
    ax[1].plot(S,mono_env(cur),'o-',color=col,lw=2.8,ms=5,label=name)
ax[0].set_title('FIXED THRESHOLD (τ=1e-3): one point per model',fontsize=11)
ax[0].set_xlabel('natural sparsity at τ=1e-3 (%)'); ax[0].set_ylabel('pruned accuracy (%)'); ax[0].set_ylim(0,90); ax[0].set_xlim(0,100)
ax[1].set_title('ENVELOPE (best acc over strength, monotone)',fontsize=11)
ax[1].set_xlabel('unstructured weight sparsity (%)'); ax[1].set_ylabel('best acc at sparsity ≥ s (%)'); ax[1].set_ylim(0,90); ax[1].set_xlim(0,95)
for a in ax: a.grid(alpha=.3); a.legend(fontsize=9,loc='lower left')
fig.suptitle('vit_tiny / CIFAR-100 — magnitude prune: L1 vs spatial vs spatial-block  [L1/spatial=regime2, block=blk_shard B16]',fontsize=11.5)
fig.tight_layout(rect=[0,0,1,.95])
out='/home/judah/spatial_v2/block_sparsity/vittiny_prune_2view.png'; fig.savefig(out,dpi=130); print('saved',out)
for name,_,ft,cur in DATA:
    print('%-28s'%name,'FT pts:',[(round(x,0),round(y,0)) for x,y in sorted(ft)])
print('\nenvelope:')
print('spars ',' '.join('%4d'%s for s in SP))
for name,_,_,cur in DATA: print('%-14s'%name[:14],' '.join('%4.0f'%v for v in mono_env(cur)))
