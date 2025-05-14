#!/bin/bash

# device mode gamma dataset_name model_name
mkdir -p ./results/$4/$2/
CUDA_VISIBLE_DEVICES=$1 python finetune.py $2 $3 $4 $5 > ./results/$4/$2/$2:$5:$3.txt
