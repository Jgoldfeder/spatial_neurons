#!/bin/bash
# Unattended ViT-base block-sparsity suite. Waits for saved base, runs all jobs across 3 GPUs, failure-tolerant.
cd /home/judah/spatial_v2/block_sparsity
PY=/opt/miniconda3/bin/python
LOG=/home/judah/spatial_v2/block_sparsity/vit_suite_logs
mkdir -p $LOG
BASE=/home/judah/spatial_v2/block_sparsity/vitbase_cifar100_base.pt

if [ ! -f "$BASE" ]; then
  echo "[orch] base missing -> finetuning ViT-base on GPU0 (blocking)..." >> $LOG/orch.log
  CUDA_VISIBLE_DEVICES=0 $PY -u /home/judah/spatial_v2/block_sparsity/vit_base_finetune.py > $LOG/00_base_finetune.log 2>&1
fi
# if still missing (finetune failed), retry once
if [ ! -f "$BASE" ]; then
  echo "[orch] base still missing -> retry finetune..." >> $LOG/orch.log
  CUDA_VISIBLE_DEVICES=0 $PY -u /home/judah/spatial_v2/block_sparsity/vit_base_finetune.py > $LOG/00_base_finetune_retry.log 2>&1
fi
until [ -f "$BASE" ]; do sleep 30; done
sleep 20
echo "[orch] base ready, starting suite at $(date +%s)" >> $LOG/orch.log

# ---- full job list (name | command) ----
JOBS=(
 "mag_contig|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py mag contig 0"
 "mag_reorder|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py mag reorder 0"
 "taylor_contig|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py taylor contig 0"
 "taylor_reorder|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py taylor reorder 0"
 "spatial_reorder_g32|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py spatial reorder 32"
 "spatial_reorder_g64|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py spatial reorder 64"
 "spatial_reorder_g128|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py spatial reorder 128"
 "spatial_reorder_g256|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py spatial reorder 256"
 "spatial_contig_g64|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py spatial contig 64"
 "glasso_reorder_0.05|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py glasso reorder 0.05"
 "glasso_reorder_0.2|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py glasso reorder 0.2"
 "glasso_reorder_0.5|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py glasso reorder 0.5"
 "glasso_reorder_1.0|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py glasso reorder 1.0"
 "glasso_contig_0.5|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit.py glasso contig 0.5"
 "movement_plain|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_movement.py plain"
 "movement_kd|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_movement.py kd"
 "rigl_0.9|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_rigl.py 0.9"
 "rigl_0.8|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_rigl.py 0.8"
 "chan_mag|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_channel.py mag"
 "chan_taylor|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_channel.py taylor"
 "chan_fpgm|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_channel.py fpgm"
 "chan_hessian|$PY -u /home/judah/spatial_v2/block_sparsity/iter_vit_channel.py hessian"
)
GPUS=(0 1 2)
declare -A PID_OF; declare -A NAME_OF
ji=0; N=${#JOBS[@]}
launch(){ local g=$1; local entry="${JOBS[$ji]}"; local name="${entry%%|*}"; local cmd="${entry#*|}"
  echo "[orch] $(date +%H:%M:%S) GPU$g <- job $ji/$N : $name" >> $LOG/orch.log
  CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > $LOG/$name.log 2>&1 &
  PID_OF[$g]=$!; NAME_OF[$g]=$name; ji=$((ji+1)); }
# initial fill
for g in ${GPUS[@]}; do [ $ji -lt $N ] && launch $g; done
# scheduler loop
while :; do
  running=0
  for g in ${GPUS[@]}; do
    p=${PID_OF[$g]}
    if [ -n "$p" ] && kill -0 $p 2>/dev/null; then running=1
    else
      [ -n "${NAME_OF[$g]}" ] && echo "[orch] $(date +%H:%M:%S) GPU$g done: ${NAME_OF[$g]}" >> $LOG/orch.log && NAME_OF[$g]=""
      if [ $ji -lt $N ]; then launch $g; running=1; fi
    fi
  done
  [ $running -eq 0 ] && [ $ji -ge $N ] && break
  sleep 20
done
echo "[orch] ALL DONE at $(date +%s)" >> $LOG/orch.log
