import torch,warnings,itertools,pickle
warnings.filterwarnings('ignore')
dev='cuda'; dt=torch.float16
M=K=4096; N=4096                     # y = W(M,K) @ x(K,N), large compute-bound GEMM
torch.manual_seed(0)
def cuda_time(fn,warm=25,it=100):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    ts=[]
    for _ in range(it):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts=sorted(ts); return ts[len(ts)//2]   # median ms
def bytes_of(t):
    b=t.values().numel()*t.values().element_size()
    for a in ['col_indices','crow_indices','row_indices','ccol_indices']:
        if hasattr(t,a):
            try: b+=getattr(t,a)().numel()*getattr(t,a)().element_size()
            except: pass
    return b
x=torch.randn(K,N,device=dev,dtype=dt)
# dense baseline
Wd=torch.randn(M,K,device=dev,dtype=dt)
dense_ms=cuda_time(lambda:Wd@x)
dense_mem=Wd.numel()*Wd.element_size()
print('DENSE fp16 %dx%d @ %dx%d : %.3f ms | %.1f MB'%(M,K,K,N,dense_ms,dense_mem/1e6),flush=True)
BLOCKS=[16,32,64,128]; SPARS=[0.0,0.5,0.7,0.8,0.9,0.95,0.99]
res={'dense':{'ms':dense_ms,'mem':dense_mem},'bsr':{},'csr':{}}
def make_blocksparse(B,p):
    W=torch.randn(M,K,device=dev,dtype=dt)
    nb=(M//B)*(K//B); Wb=W.view(M//B,B,K//B,B)
    mask=(torch.rand(M//B,K//B,device=dev)>=p)          # keep fraction (1-p)
    W=(Wb*mask[:,None,:,None]).reshape(M,K).contiguous()
    return W,float((~mask).float().mean())
def make_unstruct(p):
    W=torch.randn(M,K,device=dev,dtype=dt)
    W=W*(torch.rand(M,K,device=dev)>=p)
    return W.contiguous(),p
print('\n=== BLOCK-SPARSE (Triton bsr) ===',flush=True)
print('%4s %6s | %8s %8s %7s | %9s %7s'%('B','blk-sp','time ms','vs dense','speedup','mem MB','vs dense'),flush=True)
for B in BLOCKS:
    for p in SPARS:
        W,act=make_blocksparse(B,p)
        try:
            bsr=W.to_sparse_bsr(B)
            ms=cuda_time(lambda:(bsr@x)); mem=bytes_of(bsr)
            res['bsr'][(B,p)]={'ms':ms,'mem':mem,'act_sp':act}
            print('%4d %5.0f%% | %8.3f %7.2fx %6.2fx | %8.1f %6.2fx'%(B,act*100,ms,ms/dense_ms,dense_ms/ms,mem/1e6,mem/dense_mem),flush=True)
        except Exception as ex:
            print('%4d %5.0f%% | FAIL %s'%(B,p*100,str(ex)[:60]),flush=True)
print('\n=== UNSTRUCTURED (cuSPARSE CSR) — fair baseline ===',flush=True)
print('%6s | %8s %8s %7s | %9s %7s'%('sp','time ms','vs dense','speedup','mem MB','vs dense'),flush=True)
for p in SPARS:
    W,act=make_unstruct(p)
    try:
        csr=W.to_sparse_csr()
        ms=cuda_time(lambda:torch.sparse.mm(csr,x)); mem=bytes_of(csr)
        res['csr'][p]={'ms':ms,'mem':mem}
        print('%5.0f%% | %8.3f %7.2fx %6.2fx | %8.1f %6.2fx'%(act*100,ms,ms/dense_ms,dense_ms/ms,mem/1e6,mem/dense_mem),flush=True)
    except Exception as ex:
        print('%5.0f%% | FAIL %s'%(p*100,str(ex)[:60]),flush=True)
pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/gpubench.pkl','wb'))
print('\ndone',flush=True)
