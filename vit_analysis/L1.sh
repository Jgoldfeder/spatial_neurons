#!/bin/bash

# device
bash finetune_vit.sh $1 L1 50
bash finetune_vit.sh $1 L1 200
bash finetune_vit.sh $1 L1 500
bash finetune_vit.sh $1 L1 1000
bash finetune_vit.sh $1 L1 2000 
bash finetune_vit.sh $1 L1 4000 
bash finetune_vit.sh $1 L1 5000 
bash finetune_vit.sh $1 L1 10000 
bash finetune_vit.sh $1 L1 20000 