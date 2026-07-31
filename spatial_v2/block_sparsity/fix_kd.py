s=open('/home/judah/spatial_v2/block_sparsity/iter_movement_kd.py').read()
bad="""for e in range(4):
    m.train()
    for x,y in trl: x,y=x.to(dev),y.to(dev); _st=m(x)
        with torch.no_grad(): _tea=teacher(x)
        _T=2.0; _kd=F.kl_div(F.log_softmax(_st/_T,1),F.softmax(_tea/_T,1),reduction='batchmean')*(_T*_T)
        (F.cross_entropy(_st,y)+0.5*_kd).backward(); opt.step(); opt.zero_grad()
import copy as _c; teacher=_c.deepcopy(m); teacher.eval()
for p in teacher.parameters(): p.requires_grad=False"""
good="""for e in range(4):
    m.train()
    for x,y in trl: x,y=x.to(dev),y.to(dev); F.cross_entropy(m(x),y).backward(); opt.step(); opt.zero_grad()
import copy as _c; teacher=_c.deepcopy(m); teacher.eval()
for p in teacher.parameters(): p.requires_grad=False"""
assert bad in s, "pattern not found"
open('/home/judah/spatial_v2/block_sparsity/iter_movement_kd.py','w').write(s.replace(bad,good))
print("fixed base finetune")
