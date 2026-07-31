import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,'/home/judah/spatial_v2'); import spatial_wrapper as swc
import timm
dev='cuda'; B=15; GAMMA=32.; torch.manual_seed(0); np.random.seed(0)
T0=time.time(); P=lambda s: print('[%6.1fs] %s'%(time.time()-T0,s),flush=True)
Xtr,Ytr=torch.load('/home/judah/spatial_v2/block_sparsity/vits_cache_train.pt'); Xtr=Xtr.to(dev); Ytr=Ytr.to(dev)
Xte,Yte=torch.load('/home/judah/spatial_v2/block_sparsity/vits_cache_test.pt'); Xte=Xte.to(dev); Yte=Yte.to(dev)
P('cache loaded onto GPU %s'%(tuple(Xtr.shape),))
def norm(x): return x.float().div_(127.5).sub_(1.)
base=timm.create_model('vit_small_patch16_224',pretrained=True,num_classes=100).to(dev)
P('model created')
net=swc.SpatialCNN(base,gamma=GAMMA,device=dev,block_size=B).to(dev); m=net.model
P('SpatialCNN wrapped (%d layers)'%len(net.layers))
opt=torch.optim.AdamW([{'params':m.parameters(),'lr':1e-4}],weight_decay=0.05)
for e in range(4):
    m.train()
    ts=time.time(); net.swap(block=256); P('epoch %d SWAP done in %.1fs'%(e,time.time()-ts))
    tt=time.time(); nb=0
    perm=torch.randperm(Xtr.shape[0],device=dev)
    for i in range(0,len(perm),64):
        ix=perm[i:i+64]; x=norm(Xtr[ix]); y=Ytr[ix]
        if torch.rand(1).item()<0.5: x=torch.flip(x,[3])
        (F.cross_entropy(net(x),y)+net.get_cost()).backward(); opt.step(); opt.zero_grad(); nb+=1
        if nb==20: P('  epoch %d: 20 train steps in %.1fs (%.2fs/step)'%(e,time.time()-tt,(time.time()-tt)/20))
    P('epoch %d TRAIN (%d steps) done in %.1fs'%(e,nb,time.time()-tt))
P('DONE all 4 epochs')
