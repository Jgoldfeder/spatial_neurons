import sys,time,copy,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import timm,torchvision,torchvision.transforms as T,torch_pruning as tp
from torch.utils.data import DataLoader,Subset
METHOD=sys.argv[1]  # mag|taylor|fpgm|hessian
dev='cuda'; R=224; torch.manual_seed(0); np.random.seed(0)
tr=T.Compose([T.Resize(R),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(R),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
full=torchvision.datasets.CIFAR100('./data',train=True,transform=tr); idx=np.random.RandomState(0).permutation(len(full))[:20000]
trl=DataLoader(Subset(full,idx),64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100('./data',train=False,transform=te),256,num_workers=4)
def vit():
    m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
    m.load_state_dict(torch.load('/home/judah/spatial_v2/block_sparsity/vitbase_cifar100_base.pt',map_location='cpu')); return m.to(dev)
@torch.no_grad()
def acc(m):
    m.eval();c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev);c+=(m(x).argmax(1)==y).sum().item();t+=y.numel()
    return 100.*c/t
def finetune(m,ep,lr):
    opt=torch.optim.AdamW(m.parameters(),lr,weight_decay=0.05); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep*len(trl))
    for e in range(ep):
        m.train()
        for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); sched.step(); opt.zero_grad()
IMP={'mag':lambda:tp.importance.GroupMagnitudeImportance(p=2),'taylor':lambda:tp.importance.GroupTaylorImportance(),
     'fpgm':lambda:tp.importance.FPGMImportance(p=2),'hessian':lambda:tp.importance.GroupHessianImportance()}[METHOD]
t0=time.time(); m=vit(); pbase=sum(p.numel() for p in m.parameters()); print('base %.1f | %.1fM (%.0fs)'%(acc(m),pbase/1e6,time.time()-t0),flush=True)
ex=torch.randn(1,3,R,R).to(dev)
ign=[mm for mm in m.modules() if isinstance(mm,nn.Linear) and mm.out_features==100]
try:
    num_heads={}
    for name,mod in m.named_modules():
        if hasattr(mod,'num_heads') and hasattr(mod,'qkv'): num_heads[mod.qkv]=mod.num_heads
    pruner=tp.pruner.MetaPruner(m,ex,importance=IMP(),pruning_ratio=0.6,iterative_steps=6,ignored_layers=ign,
                               num_heads=num_heads,prune_head_dims=True,prune_num_heads=False)
except Exception as ex2:
    pruner=tp.pruner.MetaPruner(m,ex,importance=IMP(),pruning_ratio=0.6,iterative_steps=6,ignored_layers=ign)
res=[]
for step in range(6):
    if METHOD in ('taylor','hessian'):
        m.zero_grad()
        for i,(x,y) in enumerate(trl):
            F.cross_entropy(m(x.to(dev)),y.to(dev)).backward()
            if i>=10: break
    pruner.step(); finetune(m,2,5e-5)
    p=sum(pp.numel() for pp in m.parameters()); a=acc(m)
    res.append((100.*p/pbase,a)); print('[vitchan-%s] step %d | params %.1f%% | acc %.1f (%.0fs)'%(METHOD,step+1,100.*p/pbase,a,time.time()-t0),flush=True)
import pickle; pickle.dump(res,open('/home/judah/spatial_v2/block_sparsity/vit_chan_%s.pkl'%METHOD,'wb'))
print('done',flush=True)
