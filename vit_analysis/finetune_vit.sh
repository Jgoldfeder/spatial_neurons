#!/bin/bash

# device mode gamma
mkdir -p ./results/$2/
mkdir -p ./models/$2/

CUDA_VISIBLE_DEVICES=$1 python finetune_vit.py $2 $3 > ./results/$2/$2:$3.txt
