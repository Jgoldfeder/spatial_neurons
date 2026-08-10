import sys, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'/home/judah/spatial_neurons')
import spatial_wrapper_cnn as swc
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

# v4 zoo: equal-budget regularized training. One model per invocation.
#   zoo_v4.py rn18 plain 0 | rn18 l1 3000 | rn18 spatial 256
ARCH, METHOD, HP = sys.argv[1], sys.argv[2], float(sys.argv[3])
EP = int(os.environ.get('ZOO_EP','4'))
dev='cuda'; B=64; torch.manual_seed(0); np.random.seed(0)
OUT=os.environ.get('SN_OUT','/home/judah/spatial_neurons')
ZD=OUT+'/v4_zoo'; os.makedirs(ZD,exist_ok=True)
TAG='%s_%s_%g'%(ARCH,METHOD,HP)

tr=T.Compose([T.Resize(224),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
te=T.Compose([T.Resize(224),T.ToTensor(),T.Normalize((.5,.5,.5),(.5,.5,.5))])
trl=DataLoader(torchvision.datasets.CIFAR100(OUT+'/data',train=True,transform=tr,download=True),64,shuffle=True,num_workers=8,pin_memory=True)
tel=DataLoader(torchvision.datasets.CIFAR100(OUT+'/data',train=False,transform=te,download=True),256,num_workers=4)

def base_model():
    if ARCH=='vit':
        import timm; m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=100)
        m.load_state_dict(torch.load(OUT+'/vitbase_cifar100_base.pt',map_location='cpu'))
    elif ARCH=='rn50':
        m=torchvision.models.resnet50(); m.fc=nn.Linear(2048,100)
        m.load_state_dict(torch.load(OUT+'/rn50_cifar100_base.pt',map_location='cpu'))
    else:
        m=torchvision.models.resnet18(); m.fc=nn.Linear(512,100)
        m.load_state_dict(torch.load(OUT+'/rn18_cifar100_base.pt',map_location='cpu'))
    return m.to(dev)
def regl(m): return [x for _,x in m.named_modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
@torch.no_grad()
def acc(m):
    m.eval(); c=t=0
    for x,y in tel:
        x,y=x.to(dev),y.to(dev); c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return 100.*c/t
def onehot(ids,n):
    O=torch.zeros(len(ids),n,device=dev); O[torch.arange(len(ids)),torch.as_tensor(ids,device=dev,dtype=torch.long)]=1.; return O
def net_tiling(net):
    D=[]
    for li in range(len(net.layers)):
        xi,yi,xo,yo=net.planes(li)
        def ids(x,y):
            k=np.stack([x.detach().cpu().numpy(),y.detach().cpu().numpy()],1)
            _,inv=np.unique(k,axis=0,return_inverse=True); return inv
        go,gi=ids(xo,yo),ids(xi,yi)
        D.append((onehot(go,int(go.max()+1)),onehot(gi,int(gi.max()+1))))
    return D
def wmat(l): return l.weight.view(l.weight.shape[0],-1) if isinstance(l,nn.Conv2d) else l.weight
def gl_pen(lays,D):
    t=0.;n=0
    for l,(Ro,Co) in zip(lays,D):
        bn=torch.sqrt((Ro.t()@(wmat(l)**2)@Co)+1e-12); t=t+bn.sum(); n+=bn.numel()
    return t/n
def l1_pen(lays):
    t=0.;n=0
    for l in lays: t=t+l.weight.abs().sum(); n+=l.weight.numel()
    return t/n

t0=time.time()
mb=base_model(); lays=regl(mb)
net=None; GD=None
if METHOD=='spatial':
    net=swc.SpatialCNN(mb,gamma=HP,device=dev,block_size=B).to(dev); net.swap(block=256)
elif METHOD=='glasso':
    net=swc.SpatialCNN(mb,gamma=64.,device=dev,block_size=B).to(dev); net.swap(block=256)
    GD=net_tiling(net)
opt=torch.optim.AdamW(mb.parameters(),5e-5,weight_decay=0.05)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP*len(trl))
for e in range(EP):
    mb.train()
    for x,y in trl:
        x,y=x.to(dev),y.to(dev)
        loss=F.cross_entropy((net(x) if METHOD=='spatial' else mb(x)),y)
        if METHOD=='spatial': loss=loss+net.get_cost()
        elif METHOD=='l1': loss=loss+HP*l1_pen(lays)
        elif METHOD=='glasso': loss=loss+HP*gl_pen(lays,GD)
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
    if METHOD=='spatial': net.swap(block=256)
    print('[%s] epoch %d done (%.0fs)'%(TAG,e,time.time()-t0),flush=True)
a=acc(mb)
with torch.no_grad():
    allw=torch.cat([l.weight.abs().flatten() for l in lays])
    natsp=100.*(allw<1e-3).float().mean().item()
torch.save(mb.state_dict(),'%s/%s.pt'%(ZD,TAG))
if net is not None: torch.save({'pos':[p.detach().cpu() for p in net.pos]},'%s/%s_pos.pt'%(ZD,TAG))
print('[%s] FINAL dense %.2f  natsp %.1f%%  (%.0fs)'%(TAG,a,natsp,time.time()-t0),flush=True)
print('done',flush=True)
