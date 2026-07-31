import numpy as np,torch,torch.nn as nn,torchvision,sys,warnings
warnings.filterwarnings('ignore'); torch.set_grad_enabled(False)
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
B=16
def wmat(l): return l.weight.view(l.out_channels,-1) if isinstance(l,nn.Conv2d) else l.weight
def empty_at(net, thr_frac):
    # thr_frac = global element sparsity; empty-block frac on the net's CURRENT positions
    lays=net.layers
    allw=np.concatenate([wmat(l).abs().flatten().cpu().numpy() for l in lays])
    thr=np.quantile(allw,thr_frac)
    num=den=0
    for l in range(len(lays)):
        W=wmat(lays[l]).abs().cpu().numpy()
        xi,yi,xo,yo=net.planes(l)
        def ids(x,y):
            k=np.stack([x.cpu().numpy(),y.cpu().numpy()],1); _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go=ids(xo,yo); gi=ids(xi,yi); al=W>=thr
        oo=np.argsort(go,kind='stable'); ii=np.argsort(gi,kind='stable'); M=al[oo][:,ii]; g2o=go[oo]; g2i=gi[ii]
        ob=np.append(np.searchsorted(g2o,np.arange(g2o.max()+1)),len(g2o)); ib=np.append(np.searchsorted(g2i,np.arange(g2i.max()+1)),len(g2i))
        for a in range(len(ob)-1):
            ra=M[ob[a]:ob[a+1]]
            if ra.size==0: continue
            for b in range(len(ib)-1):
                bl=ra[:,ib[b]:ib[b+1]]
                if bl.size: num+=(not bl.any()); den+=1
    return 100.*num/max(den,1)
# swap g=256 model
base=torchvision.models.resnet18(); base.fc=nn.Linear(512,100)
net=swc.SpatialCNN(base,gamma=256.,device='cpu',block_size=B)
net.model.load_state_dict(torch.load('blk_models/swap_256_ep14.pt',map_location='cpu'))
# (a) TRAINED tiling: load saved positions
for p,s in zip(net.pos,torch.load('blk_models/swap_256_ep14_pos.pt',map_location='cpu')['pos']): p.copy_(s)
e_trained=empty_at(net,0.90)
print('swap g=256 @ 90%% elem-sparsity, empty-block %%:')
print('  (a) TRAINED tiling         : %.1f'%e_trained)
# (b) PROPER reorder initialized FROM trained tiling -> re-run swap (should NOT drop)
net.swap(block=256)
e_proper_fromtrained=empty_at(net,0.90)
print('  (b) proper swap-reorder     : %.1f   (init from trained; must be >= a)'%e_proper_fromtrained)
# (c) PROPER reorder from SCRATCH (fresh random positions) -> swap
net2=swc.SpatialCNN(torchvision.models.resnet18(),gamma=256.,device='cpu',block_size=B)
net2.model.fc=nn.Linear(512,100)  # fix head
net2=swc.SpatialCNN(base,gamma=256.,device='cpu',block_size=B)  # fresh random positions on same weights
net2.model.load_state_dict(torch.load('blk_models/swap_256_ep14.pt',map_location='cpu'))
net2.swap(block=256)
e_proper_scratch=empty_at(net2,0.90)
print('  (c) proper swap-reorder scratch: %.1f'%e_proper_scratch)
print('  (ref) barycenter reorder was ~15  <-- the broken one')
print('done')
