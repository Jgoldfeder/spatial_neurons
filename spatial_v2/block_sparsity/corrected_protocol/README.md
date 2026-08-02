# Corrected-protocol ViT-base block pruning

Reproduces the 2026-08-01/02 corrected-protocol experiments: iterative 16x16 block
pruning of ViT-base on CIFAR-100 with (a) a spatial pre-organization phase before the
first cut and (b) full finetune budget (6 epochs x 50k) per stage, extended to 99%
block sparsity. See `results.txt` for the full tables and findings; the original
(flawed-protocol) suite lives in the parent directory (`iter_vit.py`,
`vit_suite_orchestrator.sh`).

## Files

- `iter_vit_long.py` — main experiment script (all arms)
- `vit_base_finetune.py` — creates the dense base checkpoint (`vitbase_cifar100_base.pt`, 91.1)
- `spatial_wrapper_cnn.py` — SpatialCNN wrapper (wiring cost, swap assignment); exact copy used
- `results.txt` — final tables + findings + caveats
- `diagnostics/` — one-off checks that motivated the protocol fix:
  - `verify_ab.py` — (A) swap is function-preserving; (B) 20%-target reruns: mag vs taylor
    scoring, short vs long finetune (post-cut accs: mag 0.77 = chance, taylor 79.3)
  - `verify_b3.py` — blind mag cut + 6ep x 50k finetune (budget rebuilds 0.77 -> ~87)
  - `verify_preorg.py` — weak pre-org (2ep, gamma=64) does NOT protect the cut (1.48)

## Reproduce

```bash
# 0) base checkpoint (once, ~30 min on one GPU)
python vit_base_finetune.py                      # -> ./vitbase_cifar100_base.pt

# 1) the four arms (each ~8-10 h on one 24GB GPU; run in parallel on separate GPUs)
python iter_vit_long.py taylor  reorder 0                    # taylor baseline
python iter_vit_long.py spatial reorder 256 taylorscore      # BEST: pre-org + taylor scoring
python iter_vit_long.py spatial reorder 256                  # regular spatial (mag scoring)
python iter_vit_long.py spatial reorder 256 polish           # gamma off last 2 ft epochs
```

Args: `METHOD TIL HP [VARIANT]` — METHOD in {taylor, spatial, mag, glasso}; TIL in
{reorder, contig}; HP = gamma (spatial) / lambda (glasso); VARIANT in {'', taylorscore,
polish}. Outputs per run: `vitlong_<method><variant>_<til>_<hp>.pkl` (list of
(block-sparsity%, acc), appended per stage, crash-safe) and per-stage checkpoints in
`vitlong_models/` (plus `*_preorg.pt` and `*_pos.pt` for spatial arms). Each stage also
logs post-cut (pre-finetune) accuracy to stdout.

Protocol notes: pre-organization (spatial arms only) = 6 dense epochs with wiring cost,
re-swap each epoch, re-tile before the first cut. Stages prune to
[20,40,55,65,75,80,85,90,95,97,98,99]% of 16x16 blocks (block-L2 or taylor (g*W)^2
scoring), then finetune 6 epochs on the full 50k train set (AdamW 5e-5, cosine per
stage), masks reapplied every step. Eval = full 10k test set.

Known gaps (see results.txt): random-mask control, 2nd seed, gamma sweep, KD.
