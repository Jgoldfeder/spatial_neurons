#!/bin/bash
cd /home/judah/spatial_v2/block_sparsity
PY=/opt/miniconda3/bin/python
L=/home/judah/spatial_v2/block_sparsity/vit_suite_logs
mkdir -p $L
# wait for GPU0's current spatial-contig run to finish (its pkl appears)
until [ -f vit2_spatial_contig_256.pkl ]; do sleep 30; done
echo "[seq] spatial-contig done, running spataylor sequentially $(date +%s)" >> $L/seq.log
CUDA_VISIBLE_DEVICES=0 $PY -u /home/judah/spatial_v2/block_sparsity/iter_vit2.py spataylor reorder 256 > $L/spataylor_256.log 2>&1
echo "[seq] spataylor256 done $(date +%s)" >> $L/seq.log
CUDA_VISIBLE_DEVICES=0 $PY -u /home/judah/spatial_v2/block_sparsity/iter_vit2.py spataylor reorder 64 > $L/spataylor_64.log 2>&1
echo "[seq] spataylor64 done $(date +%s)" >> $L/seq.log
echo "[seq] ALL DONE $(date +%s)" >> $L/seq.log
